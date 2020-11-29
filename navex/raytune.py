
import os
import re
import logging
import socket

import ray
from ray.worker import global_worker

from .ray import overrides      # overrides e.g. services.get_node_ip_address
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
    local_ports = (34735, 34935, 33111, 35124)
    if 1:
        node = overrides.start(head=True, num_cpus=0, num_gpus=0, node_ip_address='127.0.0.1',
                               port=local_ports[0], redis_shard_ports=local_ports[1],
                               node_manager_port=local_ports[2], object_manager_port=local_ports[3],
                               redis_password=redis_pwd, include_dashboard=False, verbose=True)
        # node_ip_address, redis_shard_ports, gcs_server_port,
        # min_worker_port, max_worker_port, worker_port_list, memory,
        # object_store_memory, redis_max_memory, resources,
        # dashboard_host, dashboard_port, block,
        # plasma_directory, autoscaling_config, no_redirect_worker_output,
        # no_redirect_output, plasma_store_socket_name, raylet_socket_name,
        # temp_dir, java_worker_options, load_code_from_local,
        # code_search_path, system_config, lru_evict,
        # enable_object_reconstruction, metrics_export_port, log_style,
        # log_color)

        head_address = '127.0.0.1:%d' % local_ports[0]
        addr = ray.init(head_address, _redis_password=redis_pwd)
        # node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]
        # local_ports = [int(addr['redis_address'].split(':')[-1]),
        #                node_info['NodeManagerPort'], node_info['ObjectManagerPort']]

#        node_id = global_worker.core_worker.get_current_node_id()
#        node_info = [n for n in ray.nodes() if n['NodeID'] == node_id.hex()][0]
#        local_ports = [int(node.redis_address.split(':')[-1]),
#                       node_info['NodeManagerPort'], node_info['ObjectManagerPort']]
    else:
        os.system("ray start --head --include-dashboard 0 --num-cpus 0 --num-gpus 0 --port 34735 "
                  "          --node-manager-port=33111 --object-manager-port=35124 --redis-password=%s" % redis_pwd)
        ray.init('localhost:34735', _redis_password=redis_pwd)
    # these port numbers need to be unblocked on all node servers
    # seems that when starting a node it's impossible to define the head node port,
    #   seems that it still uses the one probably stored in redis
    remote_ports = local_ports

    hostname = socket.gethostname()
    if hostname and search_conf['host'] in hostname:
        # TODO: no need tunneling, just open ssh to localhost?
        ssh = None
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
            ("sbatch -c %d --export=ALL,CPUS=%d,HEAD_HOST=%s,HEAD_PORT=%d,NODE_PORT=%d,OBJ_PORT=%d,REDIS_PWD=%s "
             "$WRKDIR/navex/navex/ray/worker.sbatch") % (
            config.data.workers,
            config.data.workers,
            search_conf['host'],
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
