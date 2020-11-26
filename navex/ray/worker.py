import logging
import argparse
import time

import ray
import ray.services

from .ssh import Connection


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

    head_address = args.address
    if args.ssh_tunnel:
        # create ssh connection
        head_host, host_port = args.address.split(':')
        host_port = int(host_port)
        head_address = '127.0.0.1:%d' % host_port
        ssh = Connection(head_host, args.ssh_username or None, args.ssh_keyfile or None)

        # create tunnels to head, for redis, node and object managers
        try:
            ssh.tunnel(host_port, host_port)
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
    ray.services.get_node_ip_address = lambda x=None: '127.0.0.1'
    addr = ray.init(address=head_address, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                    log_to_driver=False, logging_level=logging.DEBUG, _redis_password=args.redis_password)
    node_info = [n for n in ray.nodes() if n['NodeID'] == addr['node_id']][0]

    # ports on which the worker is listening on
    local_ports = [int(addr['redis_address'].split(':')[-1]),
                   node_info['NodeManagerPort'],
                   node_info['ObjectManagerPort']]

    logging.info('ray worker started, ports: %s' % (local_ports,))

    time.sleep(3600)


if __name__ == '__main__':
    main()
