echo "Preparing dataset..."

INPUT_FILES=${1:-"/llm_reco_ssd/zhouyang12/data/kwai_comment/reward/0.0.1/test_all.parquet"}
DATASET_DIR=${2:-"/llm_reco/zhouyang12/.cache/Dataset/KwaiCommentJudgeCot/"}
PHOTO_DIR=${3:-"/llm_reco/zhouyang12/.cache/Photo"}
SYSTEM_PROMPT=${4:-"kwai_comment_judge_system_cot"}
NUM_SHARDS=${5:-1}
TOKENIZER_PATH=${6:-"/llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct"}
# MIN_VISUAL_TOKENS_PER_IMAGE=${8:-4}
# MAX_VISUAL_TOKENS_PER_IMAGE=${9:-1024}
# VIDEO_FPS=${10:-1.0}
# VIDEO_MIN_FRAMES=${11:-1}
# VIDEO_MAX_FRAMES=${12:-120}
# MAX_IMAGES=${13:-30}
# NUM_WORKERS=${14:-4}

mkdir -p "${DATASET_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 examples/kwai_video_eval/prepare_llm_judge_dataset.py \
    --input-files "${INPUT_FILES}" \
    --output-dir "${DATASET_DIR}" \
    --photo-dir "${PHOTO_DIR}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    --num-shards ${NUM_SHARDS} \
    --tokenizer-path "${TOKENIZER_PATH}"