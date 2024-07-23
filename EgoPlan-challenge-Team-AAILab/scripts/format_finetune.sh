# Format for finetuning model

# Multi-GPU setting if use
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Fixed Setting
PROJECT_ROOT="EgoPlan-challenge-Team-AAILab" # Need to set!!
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
cd ${PROJECT_ROOT}/src/video_llama

# Get model config
# 1. CONFIG: .yaml file name
# 2. DEVICE: GPU IDs for using (Ex. 0,1 -> Use GPU 0 & 1, 4,5,6,7 -> Use GPU 4 & 5 & 6 & 7)
# 3. NODE: Number of GPUs used
# 4. MASTER_PORT: port address (Ex. 26501)
CONFIG=$1
DEVICE=$2
NODE=$3
MASTER_PORT=$4

# Fintuning Start
CUDA_VISIBLE_DEVICES=${DEVICE} python -u -m torch.distributed.run \
--master_port=${MASTER_PORT} --nproc_per_node=${NODE} train.py \
--cfg-path ./train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT_${CONFIG}.yaml \
 2>&1 | tee -a EgoPlan-challenge-Team-AAILab/logs_finetune/finetune_egoplan_video_llama_${CONFIG}.log