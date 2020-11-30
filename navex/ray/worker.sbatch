#!/bin/bash
#SBATCH --time=0-03:59:00               # estimated execution time (~80s/epoch)
#SBATCH --mem-per-cpu=4G                # memory needed (~2Gb/core)
#SBATCH --gres=gpu:1                    # GPUs needed
####SBATCH -c $CPUS                        # CPUs needed (~9 per one gpu)  # given at command line
#SBATCH --constraint='pascal|volta'     # exclude the slowest GPUs
#SBATCH --signal=SIGUSR1@90             # wall time auto-resubmit using Lightning

## copy image data to local drive
mkdir /tmp/$SLURM_JOB_ID
trap "rm -r /tmp/$SLURM_JOB_ID; exit" TERM EXIT
tar -xf $WRKDIR/data/aachen.tar -C /tmp/$SLURM_JOB_ID

## start to process
cd $WRKDIR/navex
module load anaconda
source activate $WRKDIR/conda/envs/navex

set -x    # echo all commands for debug purposes

ssh -f -N -T -R $OBJ_M_PORT:127.0.0.1:$OBJ_M_PORT $HEAD_HOST
ssh -f -N -T -R $NODE_M_PORT:127.0.0.1:$NODE_M_PORT $HEAD_HOST
ssh -f -N -T -R $WPORT_S:127.0.0.1:$WPORT_S $HEAD_HOST

export TUNE_RESULT_DIR=$WRKDIR/navex/output
srun python -m navex.ray.worker --num-cpus=$CPUS --num-gpus=1 --ssh-tunnel --address=$HEAD_HOST:$HEAD_PORT \
                                --redis-shard-ports=$H_SHARD_PORTS --redis-password=$H_REDIS_PWD \
                                --head-object-manager-port=$H_OBJ_M_PORT --head-node-manager-port=$H_NODE_M_PORT \
                                --head-gcs-port=$H_GCS_PORT --head-raylet-port=$H_RLET_PORT \
                                --head-min-worker-port=$H_WPORT_S --head-max-worker-port=$H_WPORT_E \
                                --temp-dir=$WRKDIR/navex/logs \
                                --object-manager-port=$OBJ_M_PORT --node-manager-port=$NODE_M_PORT \
                                --min-worker-port=$WPORT_S --max-worker-port=$WPORT_E

sleep 1h
