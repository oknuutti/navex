
import os
import re
import logging

import ray
import ray.services

from .ray.ssh import Connection
from .ray.base import tune_asha
from .experiments.parser import ExperimentConfigParser, to_dict


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    logging.basicConfig(level=logging.INFO)

    os.makedirs(config.training.output, exist_ok=True)

    full_conf = to_dict(config)
    search_conf, hparams = full_conf.pop('search'), full_conf.pop('hparams')

    # start a ray cluster by creating the head, connect to it
    redis_pwd = '5241590000000000'
    if 1:
        ray.services.get_node_ip_address = lambda x: '127.0.0.1'
        addr = ray.init(num_cpus=1, num_gpus=0, log_to_driver=False, _redis_password=redis_pwd)
        node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]
        local_ports = [int(addr['redis_address'].split(':')[-1]),
                       node_info['NodeManagerPort'], node_info['ObjectManagerPort']]
    else:
        local_ports = (34735, 33111, 35124)
        os.system("ray start --head --include-dashboard 0 --num-cpus 1 --num-gpus 0 --port 34735 "
                  "          --node-manager-port=33111 --object-manager-port=35124 --redis-password=%s" % redis_pwd)
        ray.init('localhost:34735', _redis_password=redis_pwd)
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

    # search_conf['workers'] = 0

    # schedule workers
    workers = []
    for i in range(search_conf['workers']):
        out, err = ssh.exec(
            ("sbatch -c %d --export=ALL,CPUS=%d,HEAD_PORT=%d,NODE_PORT=%d,OBJ_PORT=%d,REDIS_PWD=%s "
             "$WRKDIR/navex/navex/ray/worker.sbatch") % (
            config.data.workers,
            config.data.workers,
            *remote_ports,
            redis_pwd,
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
    os.system("ray stop")

    if exception:
        raise Exception('This happended when trying to setup tune') from exception


if __name__ == '__main__':
    main()
