
import os
import re
import logging
import socket
import random
import threading
import time
import signal
from threading import Thread

import numpy as np

import ray

from .ray import overrides      # overrides e.g. services.get_node_ip_address
from .ray.ssh import Connection
from .ray.base import tune_asha
from .experiments.parser import ExperimentConfigParser, to_dict


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(config.training.output, exist_ok=True)
    full_conf = to_dict(config)

    node = RayTuneHeadNode(full_conf)
    node.start()


class RayTuneHeadNode:
    def __init__(self, full_conf):
        self.search_conf, self.hparams = full_conf.pop('search'), full_conf.pop('hparams')
        self.config = full_conf
        self.hostname = socket.gethostname()
        self.local_linux = self.hostname and self.search_conf['host'] in self.hostname
        self.redis_pwd = '5241590000000000'
        self.min_wport, self.max_wport = 10000, 10003
        self.local_ports = (34735, 34935, 33115, 35124, 36692, 29321, 28543)
        self.w_ports = tuple(range(self.min_wport, self.max_wport+1))
        self.workers = []
        self.remote_ports = None
        self.ssh = None
        self.healthy = True
        self.exception = None
        self.node_lists = None

    def start(self):
        """
        start a ray cluster by creating the head, connect to it
        """
        node = overrides.start(head=True, num_cpus=0, num_gpus=0, node_ip_address='127.0.0.1',
                               port=self.local_ports[0], redis_shard_ports='%d' % self.local_ports[1],
                               redis_password=self.redis_pwd,
                               node_manager_port=self.local_ports[2], object_manager_port=self.local_ports[3],
                               gcs_server_port=self.local_ports[4],
                               raylet_socket_name='tcp://127.0.0.1:%d' % self.local_ports[5] if not self.local_linux else None,
                               plasma_store_socket_name='tcp://127.0.0.1:%d' % self.local_ports[6] if not self.local_linux else None,
                               include_dashboard=True, verbose=True, temp_dir='/tmp/ray/', min_worker_port=self.min_wport,
                               max_worker_port=self.max_wport)

        logging.info('head node started with details: %s' % ((
                       node.address_info, {'metrics_agent_port': node.metrics_agent_port}),))
        logging.info('interfacing with python...')

        head_address = '127.0.0.1:%d' % self.local_ports[0]
        addr = ray.init(head_address, _redis_password=self.redis_pwd)
        # node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]
        # local_ports = [int(addr['redis_address'].split(':')[-1]),
        #                node_info['NodeManagerPort'], node_info['ObjectManagerPort']]

#        node_id = global_worker.core_worker.get_current_node_id()
#        node_info = [n for n in ray.nodes() if n['NodeID'] == node_id.hex()][0]
#        local_ports = [int(node.redis_address.split(':')[-1]),
#                       node_info['NodeManagerPort'], node_info['ObjectManagerPort']]

        # do this some how better
        self.remote_ports = self.local_ports

        if self.local_linux:
            # no need tunneling, just execute commands locally
            self.ssh = Connection(self.search_conf['host'])
        else:
            # ssh reverse tunnels remote_port => local_port
            self.ssh = Connection(self.search_conf['host'], self.search_conf['username'],
                                  self.search_conf['keyfile'], self.search_conf['proxy'], 19922)
            for lport, rport in zip(self.local_ports + self.w_ports, self.remote_ports + self.w_ports):
                if lport is not None:
                    rport = self.ssh.reverse_tunnel('127.0.0.1', lport, '127.0.0.1', rport)
                    logging.info('Reverse tunnel %s:%d => 127.0.0.1:%d' % (self.search_conf['host'], rport, lport))

        # create node lists so that workers won't be generated on same nodes
        self._populate_node_lists(self.search_conf['workers'])

        # schedule workers
        logging.info('scheduling %d workers...' % self.search_conf['workers'])
        for i in range(self.search_conf['workers']):
            self._schedule_worker()
        logging.info('following workers scheduled: %s' % ([w.slurm_job_id for w in self.workers],))

        if len(self.workers) == self.search_conf['workers']:
            # check if ray syncs the logs to local, if not, use ssh
            # ssh._fetch('scratch/navex/output/logs.tar', r'D:\projects\navex\output\logs.tar')

            self._register_signals()
            node = self

            def run_search():
                try:
                    tune_asha(node.search_conf, node.hparams, node.config)
                except Exception as e:
                    logging.error('Exception %s detected, terminating' % (e.__class__,))
                    node.exception = e
                if node.exception is None:
                    logging.info('TUNE FINISHED SUCCESSFULLY!')
                node.healthy = False

            # start the search in a thread so that won't block
            threading.Thread(target=run_search).start()

            try:
                while self.healthy:
                    time.sleep(1)

            except (Exception, KeyboardInterrupt) as e:
                self.exception = e

        logging.info('cleaning up...')
        if self.exception and not isinstance(self.exception, KeyboardInterrupt):
            time.sleep(900)

        # clean up
        for w in self.workers:
            out, err = self.ssh.exec("scancel %d" % w.slurm_job_id)

        del self.ssh
        os.system("ray stop")

        if self.exception:
            if isinstance(self.exception, KeyboardInterrupt):
                logging.info('Ctrl+C detected, exiting')
            else:
                raise Exception('This happended when running tune') from self.exception
        else:
            logging.info('exiting')

    def _populate_node_lists(self, n):
        cmd = 'sinfo --partition=gpu,gpushort --Node -o "%N %f"'
        incl = {'kepler', 'pascal'}

        # NODELIST AVAIL_FEATURES
        # gpu1 skl,volta,avx,avx2,avx512
        # ...
        out, err = self.ssh.exec(cmd)
        nodes = {line.split(' ')[0]
                    for line in out.split('\n')[1:]
                    if len(incl.intersection(line.split(' ')[1].split(','))) > 0}

        self.node_lists = []
        k = len(nodes)//n
        for i in range(n):
            if i < n-1:
                sub_list = random.sample(nodes, k)
                nodes = nodes - set(sub_list)
            else:
                sub_list = list(nodes)
            self.node_lists.append(sub_list)

    def _schedule_worker(self):
        worker = ScheduledWorkerNode(self.local_linux)

        if not self.local_linux:
            # create tunnels to/from non-slurm accessing machine
            worker.create_tunnels(self.ssh)

        worker.schedule_slurm_node(self, self.ssh, nodes=self.node_lists[len(self.workers)])
        if worker.slurm_job_id:
            self.workers.append(worker)

    def _register_signals(self):
        node = self

        def usr1_handler(signum, frame):  # pragma: no-cover
            # instruct worker(s) to save a checkpoint and exit
            logging.info('Scheduling additional worker...')
            node._schedule_worker()

        def term_handler(signum, frame):  # pragma: no-cover
            logging.info("TERM-signal received, terminating...")
            node.healthy = False

        signal.signal(signal.SIGUSR1, usr1_handler)
        signal.signal(signal.SIGTERM, term_handler)


class ScheduledWorkerNode:
    ports_used = set()

    @classmethod
    def reserve_port(cls, n=1):
        for _ in range(10):
            p = random.randint(20001, 2 ** 16 - 1)
            if n == 1:
                if p not in cls.ports_used:
                    cls.ports_used.add(p)
                    return p
            elif n > 1:
                ps = list(p + np.array(list(range(n))))
                if not cls.ports_used.intersection(ps):
                    return ps

        raise Exception('Seems that all ports are already used')

    def __init__(self, local_linux, max_workers=3):
        self.local_linux = local_linux
        self.max_workers = max_workers   # was worker_wport_n
        self.slurm_job_id = None

        if self.local_linux:
            self.listen_ports = [ScheduledWorkerNode.reserve_port() for _ in range(3)]  # was worker_ports
            self.worker_ports_start = ScheduledWorkerNode.reserve_port()                # was worker_wp0
        else:
            self.listen_ports = None
            self.worker_ports_start = None

    def create_tunnels(self, ssh):
        """
        :param ssh: ssh object connected to slurm login host
        :return:
        """
        # forward tunnels to contact worker nodes, local_ports => remote_ports
        # if port already used on the worker node that is later allocated, this will fail
        # TODO: to fix this problem, would need to create the forward tunnels after worker node init
        try:
            wps = ScheduledWorkerNode.reserve_port(n=self.max_workers)
            for p in wps:
                ssh.tunnel(p, p)
                logging.info('Forward tunnel 127.0.0.1:%d => %s:%d' % (p, ssh.host, p))
            self.worker_ports_start = wps[0]
        except OSError as e:
            logging.error('failed to allocate ports for worker node workers')
            raise e

        ps = [None] * 3
        for j in range(3):
            for k in range(10):
                try:
                    p = ScheduledWorkerNode.reserve_port()
                    ssh.tunnel(p, p)
                    ps[j] = p
                except OSError:
                    continue
                break
            if ps[j] is None:
                raise Exception("Can't allocate ports for forward tunnels")
            logging.info('Forward tunnel 127.0.0.1:%d => %s:%d' % (ps[j], ssh.host, ps[j]))
        self.listen_ports = ps

    def schedule_slurm_node(self, head, ssh, nodes=None):
        w_arg = '' if nodes else ('-w %s' % ','.join(nodes))

        # schedule work on a slurm node
        cmd = ("sbatch -c %d %s "
             "--export=ALL,CPUS=%d,HEAD_HOST=%s,HEAD_PORT=%d,H_SHARD_PORTS=%s,H_NODE_M_PORT=%d,H_OBJ_M_PORT=%d,"
             "H_GCS_PORT=%d,H_RLET_PORT=%d,H_OBJ_S_PORT=%d,H_WPORT_S=%d,H_WPORT_E=%d,H_REDIS_PWD=%s,"
             "NODE_M_PORT=%d,OBJ_M_PORT=%d,MEX_PORT=%d,WPORT_S=%d,WPORT_E=%d,DATADIR=%s "
             "$WRKDIR/navex/navex/ray/worker.sbatch") % (
                head.config['data']['workers'],
                w_arg,
                head.config['data']['workers'],
                head.search_conf['host'],
                *head.remote_ports,
                head.min_wport,
                head.max_wport + 1,
                head.redis_pwd,
                *self.listen_ports,
                self.worker_ports_start,
                self.worker_ports_start + self.max_workers + 1,
                head.config['data']['path'],
            )
        logging.debug('Executing command:\n%s' % cmd)
        out, err = ssh.exec(cmd)
        m = re.search(r'\d+$', out)

        if err or not m:
            logging.error('Could not schedule a worker, out: %s, err: %s' % (out, err))
        else:
            self.slurm_job_id = int(m[0])

        return self.slurm_job_id


if __name__ == '__main__':
    main()
