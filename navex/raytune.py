
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
from ray import ray_constants
from ray.dashboard import consts as dashboard_constants

from .ray import overrides      # overrides e.g. services.get_node_ip_address
from .ray.ssh import Connection
from .ray.base import tune_asha, tune_pbs
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

        if os.name == 'nt' and True:
            # for debugging on windows
            tune_asha(self.search_conf, self.hparams, self.config)
            quit()

        self.hostname = socket.gethostname()
        self.local_linux = self.hostname and self.search_conf['host'] in self.hostname
        self.redis_pwd = '5241590000000000'
        self.min_wport, self.max_wport = 10002, 10006
        self.local_ports = [34735, 34935, 33115, 35124, 36692, 29321, 28543, 10001]
        self.w_ports = tuple(range(self.min_wport, self.max_wport+1))

        def_node_cpus = {'kepler': 3, 'pascal': 4, 'volta': 5}
        node_types = list(def_node_cpus.keys()) if not self.search_conf['node_types'] \
                                                else self.search_conf['node_types'].split(',')
        node_cpus = def_node_cpus if not self.search_conf['node_type_cpus'] \
                                  else dict(zip(node_types, map(int, self.search_conf['node_type_cpus'].split(','))))
        self.node_cpus = {nt: node_cpus[nt] for nt in node_types}

        self.workers = []
        self.remote_ports = None
        self.ssh = None
        self.healthy = True
        self.exception = None
        self.node_configs = None
        self.maxmem = int(os.getenv('RAY_MAX_MEM', '0')) or None    # e.g. 8GB: 8*1024**3

    def start(self):
        """
        start a ray cluster by creating the head, connect to it
        """
        w_m, os_m, r_m = [None] * 3
        if self.maxmem is not None:
            w_m = 3 * 1024**3
            os_m, r_m = int((self.maxmem - w_m)*2/3), int((self.maxmem - w_m)/3)

        node = overrides.start(head=True, num_cpus=0, num_gpus=0, node_ip_address='127.0.0.1',
                               node_name=socket.gethostname(), port=self.local_ports[0],
                               ray_client_server_port=self.local_ports[6],
                               redis_shard_ports='%d' % self.local_ports[1], redis_password=self.redis_pwd,
                               node_manager_port=self.local_ports[2], object_manager_port=self.local_ports[3],
                               memory=w_m, object_store_memory=os_m, redis_max_memory=r_m,
                               raylet_socket_name='tcp://127.0.0.1:%d' % self.local_ports[4] if not self.local_linux else None,
                               plasma_store_socket_name='tcp://127.0.0.1:%d' % self.local_ports[5] if not self.local_linux else None,
                               temp_dir='/tmp/ray/', min_worker_port=self.min_wport,
                               include_dashboard=True, no_monitor=True, disable_usage_stats=True,
                               max_worker_port=self.max_wport)

        logging.info('starting head node with details: %s' % ((
                       node.address_info, {'metrics_agent_port': node.metrics_agent_port}),))
        logging.info('waiting 20s before interfacing with python...')
        time.sleep(20)

        head_address = '127.0.0.1:%d' % self.local_ports[0]
        addr = ray.init(head_address, _redis_password=self.redis_pwd)
        # node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]
        # local_ports = [int(addr['redis_address'].split(':')[-1]),
        #                node_info['NodeManagerPort'], node_info['ObjectManagerPort']]

#        node_id = global_worker.core_worker.get_current_node_id()
#        node_info = [n for n in ray.nodes() if n['NodeID'] == node_id.hex()][0]
#        local_ports = [int(node.redis_address.split(':')[-1]),
#                       node_info['NodeManagerPort'], node_info['ObjectManagerPort']]

        dashboard_rpc_address = ray.experimental.internal_kv._internal_kv_get(
            dashboard_constants.DASHBOARD_RPC_ADDRESS,
            namespace=ray_constants.KV_NAMESPACE_DASHBOARD,
        )
        if dashboard_rpc_address:
            logging.info("dashboard_rpc_address=%s" % (dashboard_rpc_address,))
        else:
            logging.warning("couldn't retrieve dashboard_rpc_address")
            dashboard_rpc_address = 44444
        self.local_ports.append(dashboard_rpc_address)

        # do this somehow better
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
        if self.search_conf['nodes'] > 0:
            self._populate_node_configs(self.search_conf['nodes'])

        # schedule workers
        logging.info('scheduling %d workers...' % self.search_conf['nodes'])
        for i in range(self.search_conf['nodes']):
            self._schedule_worker()
        logging.info('following workers scheduled: %s' % ([w.slurm_job_id for w in self.workers],))

        if len(self.workers) == self.search_conf['nodes']:
            # wait for first worker to have started, there seems to be a bug in ray where the processing wont start
            logging.info('waiting for the first worker to start...')
            started = False
            while not started:
                time.sleep(10)
                for w in self.workers:
                    if w.is_running(self.ssh):
                        started = True
                        break
            # wait extra 5 min so that worker has had time to launch properly
            logging.info('first worker job started, waiting extra 3 min for it to launch the ray node properly...')
            time.sleep(60 * 3)

            # check if ray syncs the logs to local, if not, use ssh
            # ssh._fetch('scratch/navex/output/logs.tar', r'D:\projects\navex\output\logs.tar')

            self._register_signals()
            node = self

            def run_search():
                try:
                    if node.search_conf['method'].split('-')[0] == 'asha':
                        tune_asha(node.search_conf, node.hparams, node.config)
                    elif node.search_conf['method'] == 'pbs':
                        tune_pbs(node.search_conf, node.hparams, node.config)
                    else:
                        assert False, 'invalid hyperparameter search method %s' % (node.search_conf['method'],)
                except Exception as e:
                    logging.error('Exception %s detected, terminating: %s' % (e.__class__, e))
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
        # if self.exception and not isinstance(self.exception, KeyboardInterrupt):
        #     time.sleep(300)

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

    def _populate_node_configs(self, n):
        cmd = 'sinfo --Node -o "%N %f %l"'
        types = set(self.node_cpus.keys())

        # NODELIST AVAIL_FEATURES
        # gpu1 skl,volta,avx,avx2,avx512
        # ...
        out, err = self.ssh.exec(cmd)
        node_list = []
        for line in out.split('\n')[1:]:
            try:
                node, feats, timelim = line.split(' ')
                feats = feats.split(',')
                feat = types.intersection(feats)
                d, time = timelim.split('-') if '-' in timelim else ('0', timelim)
                hms = time.split(':')
                if len(hms) > 2:
                    h, m, s = hms
                    timelim = int(d)*24 + int(h) + int(m)/60 + int(s)/3600
                    if len(feat) > 0 and timelim >= 12:  # in hours, should correspond to worker-*.sbatch time
                        node_list.append((list(feat)[0], node))
            except ValueError as e:
                raise Exception('Failed to parse line "%s"' % (line,)) from e

        all_nodes = set()
        grouped_nodes = {type: set() for type in self.node_cpus.keys()}
        for type, node in node_list:
            grouped_nodes[type].add(node)
            all_nodes.add(node)

        selected = set()
        self.node_configs = []
        n_handled = 0
        for type, nodes in grouped_nodes.items():
            tn = min(n - n_handled, round(n * len(nodes) / len(all_nodes)))
            n_handled += tn
            overflow = 0
            for i in range(tn):
                if i < n-1:
                    k = len(nodes) // tn
                    overflow += len(nodes) / tn - k
                    if overflow >= 1:
                        k += 1
                        overflow -= 1
                    sub_list = set(random.sample(nodes - selected, k))
                    selected.update(sub_list)
                else:
                    sub_list = nodes - selected
                self.node_configs.append((type, self.node_cpus[type], sub_list, all_nodes - sub_list))

        logging.debug('parsed node configs: %s' % (self.node_configs,))

    def _schedule_worker(self):
        worker = ScheduledWorkerNode(self.local_linux)

        if not self.local_linux:
            # create tunnels to/from non-slurm accessing machine
            worker.create_tunnels(self.ssh)

        n = len(self.node_configs)
        type, cpus, incl, excl = (None, None, None, None) if n == 0 else self.node_configs[len(self.workers) % n]
        if type is None:
            type = ','.join(self.node_cpus.keys())
        worker.schedule_slurm_node(self, self.ssh, type=type, cpus=cpus, excl_nodes=excl)
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

    @staticmethod
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                s.listen()
                return False
            except:
                return True

    @classmethod
    def reserve_port(cls, n=1):
        for _ in range(10):
            p = random.randint(20001, 2 ** 16 - 1)
            ps = [p] if n == 1 else list(p + np.array(list(range(n))))
            if not cls.ports_used.intersection(ps):
                in_use = {p for p in ps if cls.is_port_in_use(p)}
                if len(in_use) > 0:
                    cls.ports_used.update(in_use)
                    continue
                cls.ports_used.update(ps)
                return ps[0] if n == 1 else ps

        assert False, 'Seems that all ports are already used'

    def __init__(self, local_linux, max_workers=4):
        self.local_linux = local_linux
        self.max_workers = max_workers   # was worker_wport_n
        self.slurm_job_id = None

        if self.local_linux:
            self.listen_ports = [ScheduledWorkerNode.reserve_port() for _ in range(4)]  # was worker_ports
            self.worker_ports_start = ScheduledWorkerNode.reserve_port()                # was worker_wp0
        else:
            self.listen_ports = None
            self.worker_ports_start = None

    def is_running(self, ssh):
        out, err = ssh.exec(r"squeue -h -o%%t -j %s" % self.slurm_job_id)
        return out.strip().upper() == 'R'

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

        ps = [None] * 4
        for j in range(4):
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

    def schedule_slurm_node(self, head, ssh, type=None, cpus=None, excl_nodes=None):
        w_arg = '' if excl_nodes is None or len(excl_nodes) == 0 else ('-x %s' % ','.join(excl_nodes))

        # schedule work on a slurm node
        cmd = ("sbatch %s -c %d %s "
             "--export=ALL,CPUS=%d,HEAD_HOST=%s,HEAD_PORT=%d,H_SHARD_PORTS=%s,H_NODE_M_PORT=%d,H_OBJ_M_PORT=%d,"
             "H_RLET_PORT=%d,H_OBJ_S_PORT=%d,H_RCLI_PORT=%d,H_MAG_PORT=%d,H_WPORT_S=%d,H_WPORT_E=%d,"
             "H_REDIS_PWD=%s,NODE_M_PORT=%d,OBJ_M_PORT=%d,MEX_PORT=%d,MAG_PORT=%d,WPORT_S=%d,WPORT_E=%d,"
             "DATADIR=\"%s\",WRKDIR=\"$HOME/scratch\" "
             "$HOME/navex/navex/ray/worker-%s.sbatch") % (
                '' if type is None else ('-C %s' % type),
                cpus or head.config['data']['workers'],
                w_arg,
                cpus or head.config['data']['workers'],
                head.search_conf['host'],
                *head.remote_ports,
                head.min_wport,
                head.max_wport + 1,
                head.redis_pwd,
                *self.listen_ports,
                self.worker_ports_start,
                self.worker_ports_start + self.max_workers + 1,
                head.config['data']['path'],

                'triton' if head.search_conf['host'].endswith('triton.aalto.fi') else 'csc',
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
