# Format for test model

# Config Setting
MODEL_NAME="egoplan_video_llama"
PROJECT_ROOT="EgoPlan-challenge-Team-AAILab/"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
cd ${PROJECT_ROOT}

# Get model config
# 1. CONFIG: .yaml file name
# 2. DEVICE: GPU ID for testing
# 3. RAG: 'True' if your model uses RAG
# 4. EPOCH: epoch for testing
CONFIG=$1
DEVICE=$2
RAG=$3
EPOCH=$4

# If your model uses RAG, set RAG = 'True'
if [ "${RAG}" = "True" ]; then 
    MODEL_NAME="egoplan_video_llama_rag"
fi

# Test start, use 1 GPU.
CUDA_VISIBLE_DEVICES=${DEVICE} python -u eval_multiple_choice_test.py \
--model ${MODEL_NAME} \
--epic_kitchens_rgb_frame_dir Your EpicKitchens dataset path \
--ego4d_video_dir Your Ego4D dataset path \
--model_config ${CONFIG} \
--epoch ${EPOCH} \
 2>&1 | tee -a logs_test/test_multiple_choice_${CONFIG}_RAG_${RAG}_EPOCH_${EPOCH}.log