
import os

import ray

from navex.ray.ssh import Connection
from .ray.base import tune_asha
from .experiments.parser import ExperimentConfigParser, to_dict, flatten_dict


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()

    os.makedirs(config.training.output, exist_ok=True)

    full_conf = to_dict(config)
    search_conf, hparams = full_conf.pop('search'), flatten_dict(full_conf.pop('hparams'))

    # start a ray cluster by creating the head, connect to it
    addr = ray.init()
    local_host, local_port = addr['redis_address'].split(':')
    local_port = int(local_port)

    # ssh reverse tunnel  remote_port => local_port
    ssh = Connection(config.search.host, config.search.username, config.search.keyfile, config.search.proxy, 20022)
    remote_port = ssh.reverse_tunnel('', local_port)

    # schedule workers
    workers = []
    for i in range(search_conf['workers']):
        out, err = ssh.exec("sbatch --export=CPUS=%d,HEAD_ADDR='%s' $WRKDIR/navex/ray/worker.sbatch" % (
            config.data.workers,
            'triton.aalto.fi:%d' % remote_port,
        ))
        print('out:' + out)
        print('err:' + err)
        workers.append(out)

    # check if ray syncs the logs to local, if not, use ssh
    #ssh._fetch('scratch/navex/output/logs.tar', r'D:\projects\navex\output\logs.tar')

    # start the search
    # tune_asha(search_conf, hparams, full_conf)

    # clean up
    for wid in workers:
        out, err = ssh.exec("scancel %d" % (wid))
        print('out:' + out)
        print('err:' + err)

    del ssh


if __name__ == '__main__':
    main()
