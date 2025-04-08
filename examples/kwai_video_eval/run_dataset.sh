echo "Preparing dataset..."

PID_LIST_FILE=${1:-"/llm_reco_ssd/zhouyang12/data/creator/photo_ids.txt"}
DATASET_DIR=${2:-"/llm_reco/zhouyang12/.cache/Dataset/CreateBotComment/"}
PHOTO_DIR=${3:-"/llm_reco/zhouyang12/.cache/Photo"}
PROMPT=${4:-"kwai_comment_funny"}
SYSTEM_PROMPT=${5:-"default"}
NUM_SHARDS=${6:-8}
TOKENIZER_PATH=${7:-"/llm_reco/zhouyang12/models/Qwen2-VL-7B-Instruct"}
MIN_VISUAL_TOKENS_PER_IMAGE=${8:-4}
MAX_VISUAL_TOKENS_PER_IMAGE=${9:-1024}
VIDEO_FPS=${10:-1.0}
VIDEO_MIN_FRAMES=${11:-1}
VIDEO_MAX_FRAMES=${12:-120}
MAX_IMAGES=${13:-30}
NUM_WORKERS=${14:-4}

mkdir -p "${DATASET_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 tools/kwai_video/prepare_kwai_video_dataset.py \
    "${PID_LIST_FILE}" \
    --output-dir "${DATASET_DIR}" \
    --photo-dir "${PHOTO_DIR}" \
    --prompt "${PROMPT}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    --num-shards ${NUM_SHARDS} \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --min-visual-tokens-per-image ${MIN_VISUAL_TOKENS_PER_IMAGE} \
    --max-visual-tokens-per-image ${MAX_VISUAL_TOKENS_PER_IMAGE} \
    --video-fps ${VIDEO_FPS} \
    --video-min-frames ${VIDEO_MIN_FRAMES} \
    --video-max-frames ${VIDEO_MAX_FRAMES} \
    --max-images ${MAX_IMAGES} \