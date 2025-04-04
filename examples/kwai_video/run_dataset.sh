echo "Preparing dataset..."

PID_LIST_FILE=$1
DATASET_DIR=$2
PHOTO_DIR=$3
PROMPT=$4
SYSTEM_PROMPT=$5
NUM_SHARDS=$6
TOKENIZER_PATH=$7

mkdir -p "${DATASET_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 examples/kwai_video/prepare_dataset.py \
    "${PID_LIST_FILE}" \
    --output-path "${DATASET_DIR}" \
    --photo-dir "${PHOTO_DIR}" \
    --prompt "${PROMPT}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    --num-shards ${NUM_SHARDS} \
    --tokenizer-path "${TOKENIZER_PATH}"