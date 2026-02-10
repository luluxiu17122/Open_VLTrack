#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# =================================================================
# 【关键修改】添加编译时库搜索路径，解决 flashinfer 找不到 -lcuda 的问题
# 这会强制编译器去 lib/stubs 目录寻找 libcuda.so
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

#LOG_FILE="/zhdd/dataset/tyy/VOT/tyy_storage/LLM/RL/log/logfile_fulldataset_removeioureward.log"
LOG_FILE="/zssd/tyy/projects/Open_VLTrack/ReasoningTrack/Reinforcement Learning/training_grpo.log"
MODEL_PATH="/zhdd/dataset/tyy/VOT/tyy_storage/LLM/output/Qwen2.5-VL-3B-Instruct/full/train_full/checkpoint-126"  # replace it with your local file path
mkdir -p "$(dirname "$LOG_FILE")"
# MODEL_PATH="/rydata/wengchaoliu/qwen2.5vl-3b/"
#CUDA_VISIBLE_DEVICES=7 nohup python3 -m verl.trainer.main \
#    config=examples/config.yaml \
#    data.train_files=Jinliye/RLFullDataset@train \
#    data.val_files=Jinliye/RLFullDataset@test \
#    data.rollout_batch_size=256 \
#    worker.actor.model.model_path=${MODEL_PATH} \
#    worker.rollout.tensor_parallel_size=1 \
#    worker.actor.optim.strategy=adamw_bf16\
#    worker.rollout.limit_images=2 \
#    trainer.experiment_name=qwen2_5_vl_3b_tnllt_grpo \
#    trainer.n_gpus_per_node=1 \
#    trainer.logger=['console','tensorboard']\
    
    # trainer.save_checkpoint_path= "/rydata/jinliye/RL/vltracking/EasyR1/checkpoint/TNLLT" \
    # data.format_prompt= ./examples/format_prompt/LTracking_format.jinja \
    # trainer.total_epochs = 1 \
    # > /rydata/jinliye/RL/vltracking/EasyR1/log/training.log 2>&1
DATA_DIR="/zhdd/dataset/tyy/VOT/tyy_storage/LLM/RL/RLFullDataset"
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup env LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}@train \
    data.val_files=${DATA_DIR}@test \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.limit_images=2 \
    trainer.experiment_name=qwen2_5_vl_3b_tnllt_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.logger=['console','tensorboard'] > "$LOG_FILE" 2>&1 &