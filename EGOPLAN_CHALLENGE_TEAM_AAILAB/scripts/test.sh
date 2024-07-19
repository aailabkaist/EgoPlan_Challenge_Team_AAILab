# Format for evaluate model

# Naming Rule (만약 여기 있는 옵션으로만으로 사용하기 어려울 경우, 추가적으로 file을 만드는 상황의 경우, 새로 만드시고 해당 naming rule에 따라 만들어주세요.)
# Ex. eval_egoplan_video_llama_{base model}_{training dataset}_{evaluation dataset}_{specific remark (if any, ex. 10epoch)}
# {base model}_{training dataset}의 형식은 evaluation 때 해당 이름의 yaml 파일을 불러와야 하기 때문에 설정해주세요.
# model_ver은 {base model}_{training dataset}의 형식과 일치해야 합니다.

# Get model_ver and specific remark
BASE_MODEL=$1
TRAINING_DATASET=$2
DEVICE=$3
MODELSCRIPT=$4
FROMBASH=True
FIX=$5
export FIX
export FROMBASH

source ./scripts/${MODELSCRIPT}.sh
MODEL_NAME=$MODEL_NAME
PROJECT_ROOT=$PROJECT_ROOT
SPECIFIC_REMARK="None"

# Fixed Setting
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
cd ${PROJECT_ROOT}

# Evaluation dataset은 앞으로 쭉 고정이라서 따로 받지 않음
EVALUATION_DATASET="valid" # EgoPlan_validation.json 파일을 사용

PYNAME="$(python3 adapting.py pyname ${MODELSCRIPT})"
OPTION="$(python3 adapting.py option ${MODELSCRIPT})"
eval "EVAL_OPTIONS=\"${OPTION}\""

echo CUDA_VISIBLE_DEVICES=${DEVICE} nohup python3 -u ${PYNAME}.py ${EVAL_OPTIONS}

# Evaluation code, change device if needed 단일 GPU로도 충분히 돌아감.
#python3 -u ${PYNAME}.py \ ${EVAL_OPTIONS}

# Evaluation code, change device if needed 단일 GPU로도 충분히 돌아감.
CUDA_VISIBLE_DEVICES=${DEVICE} nohup bash -c "python3 -u ${PYNAME}.py ${EVAL_OPTIONS}"