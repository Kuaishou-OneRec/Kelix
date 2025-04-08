#!/bin/bash

# Make the Python script executable
chmod +x merge_bot_comments.py

# Default values
BOT_COMMENT_FILE=${1:-"/llm_reco_ssd/zhouyang12/data/creator/bot_comment.jsonl"}
RESPONSE_DIR=${2:-"/llm_reco_ssd/luoxinchen/output3/RecoVLM-Instruct/0.3.1/cmt/global_step9001_5/global_step1001/merged1001/bot_comment"}
OUTPUT_FILE=${3:-"bot_comments_sft.jsonl"}

# Run the merge script with arguments
python3 merge_bot_comments.py \
    --bot_comment_file "$BOT_COMMENT_FILE" \
    --response_dir "$RANK_DIR" \
    --output_file "$OUTPUT_FILE" 