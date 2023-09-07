EXP=$1
NAME=$2
GPUMUM=$3
set -x 

python -m torch.distributed.launch --nnodes=${GPUMUM} --nproc_per_node=8 --node_rank=$NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT main.py --name ${NAME} --base configs/${EXP}.yaml --train --logdir /mnt/data/readout_torch_output/