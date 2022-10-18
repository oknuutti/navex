import logging
import argparse
import sys
import time
import os
import signal
from subprocess import call

import ray

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
    # parser.add_argument('--head-raylet-port', type=int, help="head node raylet port")
    # parser.add_argument('--head-object-store-port', type=int, help="head node object store port")
    parser.add_argument('--head-min-worker-port', type=int, help="head node min worker port")
    parser.add_argument('--head-max-worker-port', type=int, help="head node max worker port")
    parser.add_argument('--ssh-tunnel', action="store_true", help="create tunnels to ray head")
    parser.add_argument('--ssh-username', default='', help="head redis host:port")
    parser.add_argument('--ssh-keyfile', default='', help="head redis host:port")
    parser.add_argument('--object-manager-port', type=int, help="object manager port")
    parser.add_argument('--node-manager-port', type=int, help="node manager port")
    parser.add_argument('--metrics-export-port', type=int, help="metrics export port")
    parser.add_argument('--min-worker-port', type=int, help="min worker port")
    parser.add_argument('--max-worker-port', type=int, help="max worker port")
    parser.add_argument('--maxmem', type=int, default=8*1000**3, help="max worker port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

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
            # ssh.tunnel(args.head_raylet_port, args.head_raylet_port)
            # ssh.tunnel(args.head_object_store_port, args.head_object_store_port)
            for p in range(args.head_min_worker_port, args.head_max_worker_port+1):
                ssh.tunnel(p, p)

            # create reverse tunnels from head for local node and object managers (done now in .sbatch file using ssh)
            ssh.reverse_tunnel('127.0.0.1', args.object_manager_port, '127.0.0.1', args.object_manager_port)
            ssh.reverse_tunnel('127.0.0.1', args.node_manager_port, '127.0.0.1', args.node_manager_port)
            ssh.reverse_tunnel('127.0.0.1', args.metrics_export_port, '127.0.0.1', args.metrics_export_port)
            for p in range(args.min_worker_port, args.max_worker_port+1):
                ssh.reverse_tunnel('127.0.0.1', p, '127.0.0.1', p)

        except Exception as e:
            logging.warning('ssh tunnel creation failed, maybe tunnels already exist? Exception: %s' % e)

        head_host = '127.0.0.1'

    try:
        logging.info('starting ray worker node...')
        head_address = '%s:%d' % (head_host, head_port)
        w_m, os_m, r_m = [None] * 3
        if args.maxmem is not None:
            w_m = 3 * 1024**3
            os_m, r_m = int((args.maxmem - w_m)*2/3), int((args.maxmem - w_m)/3)
        node = overrides.start(address=head_address,
                        ray_client_server_port=None, redis_password=args.redis_password,
                        object_manager_port=args.object_manager_port, node_manager_port=args.node_manager_port,
                        min_worker_port=args.min_worker_port, max_worker_port=args.max_worker_port,
                        metrics_export_port=args.metrics_export_port,
                        memory=w_m, object_store_memory=os_m, redis_max_memory=r_m,
                        num_cpus=args.num_cpus, num_gpus=args.num_gpus)

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
        _register_signals()

        while True:
            # test connection to different ports, raise exception if can't connect to a port
            _test_ports(args)
            time.sleep(15*60)   # test every 15 min

    except Exception as e:
        msg = 'Exception occurred at ray worker: %s' % e
        logging.error(msg)
        raise Exception("ray worker failed") from e


def _test_ports(args):
    fw_ports = [(name, int(getattr(args, name, 0) or 0)) for name in (
        'redis_shard_ports', 'head_object_manager_port', 'head_node_manager_port', 'head_gcs_port')]
#        , 'head_raylet_port', 'head_object_store_port')]
    fw_ports.append(('head_port', int(args.address.split(':')[1])))
    for i, p in enumerate(range(args.min_worker_port, args.max_worker_port + 1)):
        fw_ports.append(('worker_port_%d' % i, p))

    rw_ports = [(name, int(getattr(args, name, 0) or 0)) for name in
                ('object_manager_port', 'node_manager_port', 'metrics_export_port')]
    for i, p in enumerate(range(args.head_min_worker_port, args.head_max_worker_port + 1)):
        rw_ports.append(('head_worker_port_%d' % i, p))

    for name, port in fw_ports:
        if port:
            if not _test_fw_port(port):
                raise Exception("Can't connect to forwarded port %d (%s)" % (port, name))
        else:
            logging.warning('port for %s not given' % name)

    for name, port in rw_ports:
        if port:
            if not _test_rw_port(port):
                raise Exception("Can't connect to reverse fw port %d (%s)" % (port, name))
        else:
            logging.warning('port for %s not given' % name)


def _test_fw_port(port):
    # TODO: implement this, should be easy
    return True


def _test_rw_port(port):
    # TODO: implement this (using ssh? need actual head host address, not 127.0.0.1)
    return True


def _register_signals():
    """ call this after worker node init """

    # signal = Signal.options(name="term_" + node_id).remote()      # OPTIONAL?

    def sig_handler(signum, frame):  # pragma: no-cover
        # instruct worker(s) to save a checkpoint and exit
        logging.info('handling SIGUSR1')
        # signal.set.remote()   # OPTIONAL?

        # find job id
        job_id = os.environ['SLURM_JOB_ID']
        cmd = ['scontrol', 'requeue', job_id]

        # requeue job
        logging.info(f'requeing job {job_id}...')
        result = call(cmd)

        # print result text
        if result == 0:
            logging.info(f'requeued exp {job_id}')
        else:
            logging.warning('requeue failed...')

        # kill all connections first, otherwise head will try to restore trials leading to failure, then
        # a failed attempt to restart the trials, then starting the same trial from scratch on a new node
        os.system(r"pgrep -a ^ssh$")
        os.system(r"cat ~/slurm-%s.out >> ~/slurm-%s.hist" % (job_id, job_id))
        time.sleep(5)

        # worker node would use atexit in an attempt to shut itself down gracefully
        #  - however at least in v1.1.0 fails to do so, thus commented out
        #    - seems that need to exit, otherwise job requeue fails?
        sys.exit()

    def term_handler(signum, frame):  # pragma: no-cover
        logging.info("bypassing sigterm")

    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


if __name__ == '__main__':
    main()
