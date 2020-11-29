import logging
import argparse
import time

import ray

from .ssh import Connection
from . import overrides         # overrides e.g. services.get_node_ip_address


def main():
    parser = argparse.ArgumentParser('Start a custom ray worker')
    parser.add_argument('--num-cpus', type=int, help="number of cpus to reserve")
    parser.add_argument('--num-gpus', type=int, help="number of gpus to reserve")
    parser.add_argument('--ssh-tunnel', action="store_true", help="create tunnels to ray head")
    parser.add_argument('--address', help="head redis host:port")
    parser.add_argument('--ssh-username', default='', help="head redis host:port")
    parser.add_argument('--ssh-keyfile', default='', help="head redis host:port")
    parser.add_argument('--redis-password', help="head redis password")
    parser.add_argument('--object-manager-port', type=int, help="head object manager port")
    parser.add_argument('--node-manager-port', type=int, help="head node manager port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    head_host, head_port = args.address.split(':')
    head_port = int(head_port)

    if args.ssh_tunnel:
        # create ssh connection
        ssh = Connection(head_host, args.ssh_username or None, args.ssh_keyfile or None)
        head_host = '127.0.0.1'

        # create tunnels to head, for redis, node and object managers
        try:
            ssh.tunnel(head_port, head_port)
            ssh.tunnel(args.object_manager_port, args.object_manager_port)
            ssh.tunnel(args.node_manager_port, args.node_manager_port)
        except Exception as e:
            logging.warning('ssh tunnel creation failed, maybe tunnels already exist? Exception: %s' % e)

        if 0:
            # create reverse tunnels so that head can connect to worker
            worker_redis_port = ssh.reverse_tunnel('127.0.0.1', 23010)
            worker_object_port = ssh.reverse_tunnel('127.0.0.1', 23020)
            worker_node_port = ssh.reverse_tunnel('127.0.0.1', 23030)
            # TODO: how to set these ports for the worker?

    logging.info('starting ray worker...')
    head_address = '%s:%d' % (head_host, head_port)
    overrides.start(address=head_address, redis_password=args.redis_password,
                    num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                    verbose=True, include_dashboard=False)
              # node_ip_address, redis_shard_ports, object_manager_port, node_manager_port, gcs_server_port,
              # min_worker_port, max_worker_port, worker_port_list, memory,
              # object_store_memory, redis_max_memory, resources,
              # dashboard_host, dashboard_port, block,
              # plasma_directory, autoscaling_config, no_redirect_worker_output,
              # no_redirect_output, plasma_store_socket_name, raylet_socket_name,
              # temp_dir, java_worker_options, load_code_from_local,
              # code_search_path, system_config, lru_evict,
              # enable_object_reconstruction, metrics_export_port, log_style,
              # log_color)

    addr = ray.init(address=head_address, logging_level=logging.DEBUG, _redis_password=args.redis_password)
    node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]

    # ports on which the worker is listening on
    local_ports = [int(addr['redis_address'].split(':')[-1]),
                   node_info['NodeManagerPort'],
                   node_info['ObjectManagerPort']]

    logging.info('ray worker started, ports: %s' % (local_ports,))

    time.sleep(3600)


if __name__ == '__main__':
    main()
