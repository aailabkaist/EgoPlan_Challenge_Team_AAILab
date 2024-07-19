# Format for evaluate model

# Fixed Setting
MODEL_NAME="egoplan_video_llama"
PROJECT_ROOT="EgoPlan-challenge-Team-AAILab"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
cd ${PROJECT_ROOT}

# Get model_ver and specific remark
BASE_MODEL=$1
TRAINING_DATASET=$2
DEVICE=$3
SPECIFIC_REMARK="None"
# Evaluation dataset은 앞으로 쭉 고정이라서 따로 받지 않음
EVALUATION_DATASET="valid" # EgoPlan_validation.json 파일을 사용

# Evaluation code, change device if needed 단일 GPU로도 충분히 돌아감.
CUDA_VISIBLE_DEVICES=${DEVICE} nohup python3 -u eval_multiple_choice_default.py \
--model ${MODEL_NAME} \
--epic_kitchens_rgb_frame_dir Your EpicKitchens Directory \
--ego4d_video_dir Your Ego4D Directory \
--model_ver ${BASE_MODEL}_${TRAINING_DATASET} \
--save_name ${BASE_MODEL}_${TRAINING_DATASET}_${EVALUATION_DATASET}_${SPECIFIC_REMARK} \
> logs_eval/eval_multiple_choice_${MODEL_NAME}_${BASE_MODEL}_${TRAINING_DATASET}_${EVALUATION_DATASET}_${SPECIFIC_REMARK}.log 2>&1 & 
