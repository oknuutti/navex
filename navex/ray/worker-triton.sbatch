#!/bin/bash
#SBATCH --time=0-11:59:00        # estimated execution time (~80s/epoch)
#SBATCH --mem-per-cpu=2560M        # memory needed (~2Gb/core)
#SBATCH --gres=gpu:1            # GPUs needed
#SBATCH --constraint='volta'    # 'kepler|pascal|volta'
#SBATCH --signal=SIGUSR1@120    # wall time auto-resubmit

## copy image data to local drive
## set locations

set -x    # echo all commands for debug purposes

export NAVEX_DATA="/tmp/navex"
mkdir -p "$NAVEX_DATA"
trap "rm -r $NAVEX_DATA; exit" TERM EXIT

export WORKER_LOG_PATH="$WRKDIR/navex/cache/$SLURM_JOB_ID"
rm -fR /tmp/ray
rm -fR "$WORKER_LOG_PATH-old"
[ ! -d $WORKER_LOG_PATH ] || mv $WORKER_LOG_PATH "$WORKER_LOG_PATH-old"
mkdir -p $WORKER_LOG_PATH
ln -s $WORKER_LOG_PATH /tmp/ray

set +x

DATADIR=${DATADIR//;/$'\n'}
for DDIR in $DATADIR
do
  if [[ "$DDIR" == *.gz ]]
  then
    set -x
    tar -x --use-compress-program=zstdmt -f "$WRKDIR/$DDIR" -C "$NAVEX_DATA"
    set +x
  else
    set -x
    tar -xf "$WRKDIR/$DDIR" -C "$NAVEX_DATA"
    set +x
  fi
done

## start to process
cd "$WRKDIR/navex"
module load anaconda
source activate "$WRKDIR/conda/envs/navex"

H_WPORT_TUNNELS=""
for ((p=H_WPORT_S;p<=H_WPORT_E;p++)); do
  H_WPORT_TUNNELS="$H_WPORT_TUNNELS -L $p:127.0.0.1:$p"
done

WPORT_TUNNELS=""
for ((p=WPORT_S;p<=WPORT_E;p++)); do
  WPORT_TUNNELS="$WPORT_TUNNELS -R $p:127.0.0.1:$p"
done

set -x    # echo all commands for debug purposes

# forward and reverse tunnels
ssh -f -N -T -L "$HEAD_PORT:127.0.0.1:$HEAD_PORT" \
             -L "$H_SHARD_PORTS:127.0.0.1:$H_SHARD_PORTS" \
             -L "$H_OBJ_M_PORT:127.0.0.1:$H_OBJ_M_PORT" \
             -L "$H_NODE_M_PORT:127.0.0.1:$H_NODE_M_PORT" \
             -L "$H_GCS_PORT:127.0.0.1:$H_GCS_PORT" \
             $H_WPORT_TUNNELS \
             -R "$NODE_M_PORT:127.0.0.1:$NODE_M_PORT" \
             -R "$OBJ_M_PORT:127.0.0.1:$OBJ_M_PORT" \
             -R "$MEX_PORT:127.0.0.1:$MEX_PORT" \
             $WPORT_TUNNELS \
             "$HEAD_HOST"

#             maybe don't need tunnel for $H_WPORT_S, haven't tried
#             -L $H_RLET_PORT:127.0.0.1:$H_RLET_PORT \
#             -L $H_OBJ_S_PORT:127.0.0.1:$H_OBJ_S_PORT \

export TUNE_RESULT_DIR="$WRKDIR/navex/output"
srun python -m navex.ray.worker --num-cpus="$CPUS" --num-gpus=1 --address="127.0.0.1:$HEAD_PORT" \
                                --redis-shard-ports="$H_SHARD_PORTS" --redis-password="$H_REDIS_PWD" \
                                --head-object-manager-port="$H_OBJ_M_PORT" --head-node-manager-port="$H_NODE_M_PORT" \
                                --head-gcs-port="$H_GCS_PORT" \
                                --head-min-worker-port="$H_WPORT_S" --head-max-worker-port="$H_WPORT_E" \
                                --object-manager-port="$OBJ_M_PORT" --node-manager-port="$NODE_M_PORT" \
                                --metrics-export-port="$MEX_PORT" \
                                --min-worker-port="$WPORT_S" --max-worker-port="$WPORT_E" \
                                --maxmem=8000000000
#                                --head-raylet-port=$H_RLET_PORT --head-object-store-port=$H_OBJ_S_PORT \

#sleep 1h
