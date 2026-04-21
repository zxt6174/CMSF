DATASET_NAME='f30k'
DATA_PATH='/data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 infer-time-visual.py \
  --data_path ${DATA_PATH} \
  --dataset ${DATASET_NAME} \
  --trained_time 
  
  
