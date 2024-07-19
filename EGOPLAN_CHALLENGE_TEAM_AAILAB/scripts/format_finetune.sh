# Format for finetune model

# Multi-GPU setting if use
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Fixed Setting
PROJECT_ROOT="EgoPlan-challenge-Team-AAILab"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
cd ${PROJECT_ROOT}/src/video_llama

# Get base model and training dataset
BASE_MODEL=$1
TRAINING_DATASET=$2
DEVICE=$3
NODE=$4
MASTER_PORT=$5

# Finetune code, GPU 개수 바꿀 시 --nproc_per_node도 바꿔주세요. master_port의 경우도 이미 있으면 충돌하기 때문에 바꿔주셔야 합니다.
CUDA_VISIBLE_DEVICES=${DEVICE} nohup python -u -m torch.distributed.run \
--master_port=${MASTER_PORT} --nproc_per_node=${NODE} train.py \
--cfg-path ./train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT_${BASE_MODEL}_${TRAINING_DATASET}.yaml \
> ${PROJECT_ROOT}/logs_finetune/finetune_egoplan_video_llama_${BASE_MODEL}_${TRAINING_DATASET}.log 2>&1 &