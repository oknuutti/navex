
import os
import re
import logging

import ray

from navex.ray.ssh import Connection
from .ray.base import tune_asha
from .experiments.parser import ExperimentConfigParser, to_dict, flatten_dict


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    logging.basicConfig(level=logging.INFO)

    os.makedirs(config.training.output, exist_ok=True)

    full_conf = to_dict(config)
    search_conf, hparams = full_conf.pop('search'), full_conf.pop('hparams')

    # start a ray cluster by creating the head, connect to it
    addr = ray.init(num_cpus=1, num_gpus=0, log_to_driver=False)
    node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]
    local_ports = [int(addr['redis_address'].split(':')[-1]),
                   node_info['NodeManagerPort'], node_info['ObjectManagerPort']]
    # TODO: fix this: magical port numbers, other similar seem to be blocked by triton firewall
    remote_ports = (34735, 33111, 35124)

    hostname = os.getenv('HOSTNAME')
    if hostname and search_conf['host'] in hostname:
        # TODO: no need tunneling, just open ssh to localhost
        raise NotImplemented()
    else:
        # ssh reverse tunnels remote_port => local_port
        ssh = Connection(config.search.host, config.search.username, config.search.keyfile, config.search.proxy, 20022)
        for lport, rport in zip(local_ports, remote_ports):
            if lport is not None:
                rport = ssh.reverse_tunnel('127.0.0.1', lport, search_conf['host'], rport)
                logging.info('Reverse tunnel %s:%d => 127.0.0.1:%d' % (search_conf['host'], rport, lport))

    #search_conf['workers'] = 0

    # schedule workers
    workers = []
    for i in range(search_conf['workers']):
        out, err = ssh.exec(
            ("sbatch -c %d --export=ALL,CPUS=%d,HEAD_ADDR='%s',NODE_PORT=%d,OBJ_PORT=%d "
             "$WRKDIR/navex/navex/ray/worker.sbatch") % (
            config.data.workers,
            config.data.workers,
            '%s:%d' % (search_conf['host'], remote_ports[0]),
            *remote_ports[1:],
        ))
        m = re.search(r'\d+$', out)
        if err or not m:
            logging.error('Could not schedule a worker, out: %s, err: %s' % (out, err))
        else:
            workers.append(int(m[0]))

    logging.info('following workers scheduled: %s' % (workers,))

    exception = None
    if len(workers) == search_conf['workers']:
        # check if ray syncs the logs to local, if not, use ssh
        # ssh._fetch('scratch/navex/output/logs.tar', r'D:\projects\navex\output\logs.tar')

        # start the search
        try:
            tune_asha(search_conf, hparams, full_conf)
        except Exception as e:
            exception = e

    logging.info('cleaning up...')

    # clean up
    for wid in workers:
        out, err = ssh.exec("scancel %d" % wid)

    del ssh

    if exception:
        raise Exception('This happended when trying to setup tune') from exception


if __name__ == '__main__':
    main()
