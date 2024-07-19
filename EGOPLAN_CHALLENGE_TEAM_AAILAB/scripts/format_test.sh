# Format for evaluate model

# Naming Rule (만약 여기 있는 옵션으로만으로 사용하기 어려울 경우, 추가적으로 file을 만드는 상황의 경우, 새로 만드시고 해당 naming rule에 따라 만들어주세요.)
# Ex. eval_egoplan_video_llama_{base model}_{training dataset}_{evaluation dataset}_{specific remark (if any, ex. 10epoch)}
# {base model}_{training dataset}의 형식은 evaluation 때 해당 이름의 yaml 파일을 불러와야 하기 때문에 설정해주세요.
# model_ver은 {base model}_{training dataset}의 형식과 일치해야 합니다.

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
CUDA_VISIBLE_DEVICES=${DEVICE} nohup python3 -u eval_multiple_choice_test.py \
--model ${MODEL_NAME} \
--epic_kitchens_rgb_frame_dir /home/aailab/data4/kasong13/EPIC-KITCHENS \
--ego4d_video_dir /home/aailab/data4/kasong13/Ego4D/v1/full_scale \
--model_ver ${BASE_MODEL}_${TRAINING_DATASET} \
--save_name ${BASE_MODEL}_${TRAINING_DATASET}_${EVALUATION_DATASET}_${SPECIFIC_REMARK} \
> logs_test/test_multiple_choice_${MODEL_NAME}_${BASE_MODEL}_${TRAINING_DATASET}_${EVALUATION_DATASET}_${SPECIFIC_REMARK}.log 2>&1 & # full log file 저장용, save_name은 accuracy.txt 파일 생성에 쓰임
