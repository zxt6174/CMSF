DATASET_NAME='coco'
DATA_PATH='/data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=22222 train.py \
  --data_path ${DATA_PATH} \
  --data_name ${DATASET_NAME} \
  --num_epochs= \
  --lr_update= \
  --learning_rate= \
  --precomp_enc_type basic \
  --workers  \
  --log_step  \
  --embed_size  \
  --batch_size  \
  --attention_lamda  \
  --use_moco  \
  --loss_lamda  \
  --mu \
  --gama  \
  --moco_r  
