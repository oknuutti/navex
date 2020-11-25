#!/bin/bash
#SBATCH --time=0-03:59:00               # estimated execution time (~80s/epoch)
#SBATCH --mem-per-cpu=4G                # memory needed (~2Gb/core)
#SBATCH --gres=gpu:1                    # GPUs needed
####SBATCH -c $CPUS                        # CPUs needed (~9 per one gpu)  # given at command line
#SBATCH --constraint='pascal|volta'     # exclude the slowest GPUs
#SBATCH --signal=SIGUSR1@90             # wall time auto-resubmit using Lightning

set -x    # echo all commands for debug purposes

## copy image data to local drive
mkdir /tmp/$SLURM_JOB_ID
trap "rm -r /tmp/$SLURM_JOB_ID; exit" TERM EXIT
tar -xf $WRKDIR/data/aachen.tar -C /tmp/$SLURM_JOB_ID

## start to process
cd $WRKDIR/navex
module load anaconda
source activate $WRKDIR/conda/envs/navex

ssh -o ExitOnForwardFailure=yes -f -N -L $HEAD_PORT:localhost:$HEAD_PORT -L $OBJ_PORT:localhost:$OBJ_PORT \
                                      -L $NODE_PORT:localhost:$NODE_PORT triton.aalto.fi

sleep 3s

export TUNE_RESULT_DIR=$WRKDIR/navex/output
srun ray start --address=localhost:$HEAD_PORT --object-manager-port=$OBJ_PORT --redis-password=$REDIS_PWD \
               --node-manager-port=$NODE_PORT --verbose --block --num-cpus=$CPUS --num-gpus=1

sleep 1h
