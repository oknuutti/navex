
import os
import re

import ray

from navex.ray.ssh import Connection
from .ray.base import tune_asha
from .experiments.parser import ExperimentConfigParser, to_dict, flatten_dict


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()

    os.makedirs(config.training.output, exist_ok=True)

    full_conf = to_dict(config)
    search_conf, hparams = full_conf.pop('search'), full_conf.pop('hparams')

    # import ray.tune as tune
    # from ray.tune.suggest.variant_generator import generate_variants
    # print(hparams)
    # print(list(generate_variants(hparams)))
    # quit()

    # start a ray cluster by creating the head, connect to it
    addr = ray.init(num_cpus=1, num_gpus=0)
    local_host, local_port = addr['redis_address'].split(':')
    local_port = int(local_port)

    # ssh reverse tunnel  remote_port => local_port
    ssh = Connection(config.search.host, config.search.username, config.search.keyfile, config.search.proxy, 20022)
    remote_port = ssh.reverse_tunnel('', local_port)

    # schedule workers
    workers = []
    for i in range(search_conf['workers']):
        out, err = ssh.exec("sbatch -c %d --export=ALL,CPUS=%d,HEAD_ADDR='%s' $WRKDIR/navex/navex/ray/worker.sbatch" % (
            config.data.workers,
            config.data.workers,
            '%s:%d' % (search_conf['host'], remote_port),
        ))
        m = re.search(r'\d+$', out)
        if err or not m:
            print('out: ' + out)
            print('err: ' + err)
        else:
            workers.append(int(m[0]))

    print('following workers scheduled: %s' % (workers,))

    exception = None
    if len(workers) == search_conf['workers']:
        # check if ray syncs the logs to local, if not, use ssh
        # ssh._fetch('scratch/navex/output/logs.tar', r'D:\projects\navex\output\logs.tar')

        # start the search
        try:
            tune_asha(search_conf, hparams, full_conf)
        except Exception as e:
            exception = e

    print('cleaning up...')

    # clean up
    for wid in workers:
        out, err = ssh.exec("scancel %d" % wid)

    del ssh

    if exception:
        raise Exception('This happended when trying to setup tune') from exception


if __name__ == '__main__':
    main()
