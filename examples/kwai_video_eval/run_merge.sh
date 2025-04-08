#!/bin/bash

# Make the Python scripts executable
chmod +x merge_bot_comments.py
chmod +x jsonl_to_excel.py

# Default values
BOT_COMMENT_FILE=${1:-"/llm_reco_ssd/zhouyang12/data/creator/bot_comment.jsonl"}
RESPONSE_DIR=${2:-"/llm_reco_ssd/luoxinchen/output3/RecoVLM-Instruct/0.3.1/cmt/global_step9001_5/global_step1001/merged1001/bot_comment"}
OUTPUT_FILE=${3:-"bot_comments_sft.jsonl"}
NUM_RESPONSES=${4:-5}

# Run the merge script
python3 merge_bot_comments.py \
    --bot_comment_file "$BOT_COMMENT_FILE" \
    --response_dir "$RESPONSE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --num_responses "$NUM_RESPONSES"

# Convert the merged JSONL to Excel
EXCEL_FILE="${OUTPUT_FILE%.*}.xlsx"
python3 jsonl_to_excel.py \
    --input_file "$OUTPUT_FILE" \
    --output_file "$EXCEL_FILE" \
    --sheet_name "Bot Comments" \