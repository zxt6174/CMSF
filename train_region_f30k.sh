DATASET_NAME='f30k'
DATA_PATH='/data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=56667 train.py \
  --data_path ${DATA_PATH} \
  --data_name ${DATASET_NAME} \
  --num_epochs= \
  --warmup_epochs= \
  --lr_min= \
  --learning_rate= \
  --precomp_enc_type  \
  --workers  \
  --log_step  \
  --embed_size  \
  --batch_size \
  --attention_lamda  \
  --use_moco  \
  --moco_M  \
  --loss_lamda  \
  --mu  \
  --gama  \
  --moco_r  

