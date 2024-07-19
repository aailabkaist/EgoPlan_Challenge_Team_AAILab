# Format for finetune model

# Naming Rule (만약 여기 있는 옵션으로만으로 사용하기 어려울 경우, 추가적으로 file을 만드는 상황의 경우, 새로 만드시고 해당 naming rule에 따라 만들어주세요.)
# Ex. finetune_egoplan_video_llama_{model_ver}_{training dataset}_{evaluation dataset}_{specific remark (if any, ex. 10epoch)}
# {base model}_{training dataset}의 형식은 evaluation 때 해당 이름의 yaml 파일을 불러와야 하기 때문에 설정해주세요.
# model_ver은 {base model}_{training dataset}의 형식과 일치해야 합니다.

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
OPTION=$6

python using_parser.py -b ./train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT_${BASE_MODEL}_${TRAINING_DATASET}.yaml ${OPTION}&&
# Finetune code, GPU 개수 바꿀 시 --nproc_per_node도 바꿔주세요. master_port의 경우도 이미 있으면 충돌하기 때문에 바꿔주셔야 합니다.
CUDA_VISIBLE_DEVICES=${DEVICE} nohup python -u -m torch.distributed.run \
--master_port=${MASTER_PORT} --nproc_per_node=${NODE} train.py \
--cfg-path ./train_configs/visionbranch_stage3_finetune_on_EgoPlan_IT_${BASE_MODEL}_${TRAINING_DATASET}_mod.yaml \
> /home/aailab/data2/kasong13/EgoPlan-challenge/logs_finetune/finetune_egoplan_video_llama_${BASE_MODEL}_${TRAINING_DATASET}.log 2>&1 &