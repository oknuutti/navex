import logging
import argparse
import time
import os

import ray

from navex.ray.base import register_slurm_signal_handlers
from .ssh import Connection
from . import overrides         # overrides e.g. services.get_node_ip_address


def main():
    parser = argparse.ArgumentParser('Start a custom ray worker')
    parser.add_argument('--num-cpus', type=int, help="number of cpus to reserve")
    parser.add_argument('--num-gpus', type=int, help="number of gpus to reserve")
    parser.add_argument('--address', help="head redis host:port")
    parser.add_argument('--redis-shard-ports', help="redis shard ports")
    parser.add_argument('--redis-password', help="head redis password")
    parser.add_argument('--head-object-manager-port', type=int, help="head object manager port")
    parser.add_argument('--head-node-manager-port', type=int, help="head node manager port")
    parser.add_argument('--head-gcs-port', type=int, help="head node gcs port")
    parser.add_argument('--head-raylet-port', type=int, help="head node raylet port")
    parser.add_argument('--head-object-store-port', type=int, help="head node object store port")
    parser.add_argument('--head-min-worker-port', type=int, help="head node min worker port")
    parser.add_argument('--head-max-worker-port', type=int, help="head node max worker port")
    parser.add_argument('--ssh-tunnel', action="store_true", help="create tunnels to ray head")
    parser.add_argument('--ssh-username', default='', help="head redis host:port")
    parser.add_argument('--ssh-keyfile', default='', help="head redis host:port")
    parser.add_argument('--object-manager-port', type=int, help="object manager port")
    parser.add_argument('--node-manager-port', type=int, help="node manager port")
    parser.add_argument('--min-worker-port', type=int, help="min worker port")
    parser.add_argument('--max-worker-port', type=int, help="max worker port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    head_host, head_port = args.address.split(':')
    head_port = int(head_port)

    if args.ssh_tunnel:
        # create ssh connection
        ssh = Connection(head_host, args.ssh_username or None, args.ssh_keyfile or None)
        try:
            # TODO: do these work or not?

            # create tunnels to head, for redis, node and object managers
            ssh.tunnel(head_port, head_port)
            ssh.tunnel(int(args.redis_shard_ports), int(args.redis_shard_ports))
            ssh.tunnel(args.head_object_manager_port, args.head_object_manager_port)
            ssh.tunnel(args.head_node_manager_port, args.head_node_manager_port)
            ssh.tunnel(args.head_gcs_port, args.head_gcs_port)
            ssh.tunnel(args.head_raylet_port, args.head_raylet_port)
            ssh.tunnel(args.head_object_store_port, args.head_object_store_port)
            for p in range(args.head_min_worker_port, args.head_max_worker_port+1):
                ssh.tunnel(p, p)

            # create reverse tunnels from head for local node and object managers (done now in .sbatch file using ssh)
            ssh.reverse_tunnel('127.0.0.1', args.object_manager_port, '127.0.0.1', args.object_manager_port)
            ssh.reverse_tunnel('127.0.0.1', args.node_manager_port, '127.0.0.1', args.node_manager_port)
            for p in range(args.min_worker_port, args.max_worker_port+1):
                ssh.reverse_tunnel('127.0.0.1', p, '127.0.0.1', p)

        except Exception as e:
            logging.warning('ssh tunnel creation failed, maybe tunnels already exist? Exception: %s' % e)

        head_host = '127.0.0.1'

    try:
        logging.info('starting ray worker node...')
        head_address = '%s:%d' % (head_host, head_port)
        node = overrides.start(address=head_address, redis_password=args.redis_password,
                        object_manager_port=args.object_manager_port, node_manager_port=args.node_manager_port,
                        min_worker_port=args.min_worker_port, max_worker_port=args.max_worker_port,
                        num_cpus=args.num_cpus, num_gpus=args.num_gpus, verbose=True, include_dashboard=False)

        logging.info('ray worker node started with details: %s' % ((
                      node.address_info, {'metrics_agent_port': node.metrics_agent_port}),))
        logging.info('interfacing with python...')

        addr = ray.init(address=head_address, logging_level=logging.DEBUG,
                        _redis_password=args.redis_password)
        node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]

        # ports on which the worker is listening on
        local_ports = [int(addr['redis_address'].split(':')[-1]),
                       node_info['NodeManagerPort'],
                       node_info['ObjectManagerPort']]

        logging.info('ray worker node successfully initialized, ports: %s' % (local_ports,))

        # hook signals
        register_slurm_signal_handlers(addr['node_id'])

        while True:
            time.sleep(100000)
    except Exception as e:
        msg = 'Exception occurred during ray worker startup: %s' % e
        logging.error(msg)
        raise Exception("ray worker startup failed") from e


if __name__ == '__main__':
    main()
