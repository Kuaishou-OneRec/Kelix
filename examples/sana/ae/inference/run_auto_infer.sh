#!/bin/bash

# 设置参数
PYTHON_SCRIPT="/mmu_mllm_hdd_2/liziming00/eval_muse_gen/muse/muse/tools/auto_infer.py"

# FOLDERS="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp6/step3000/global_step3000"
# CACHE_TOKEN_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp6/step3000"
# CACHE_IMAGE_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp6/step3000"

# FOLDERS="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp7/step10500/global_step10500"
# CACHE_TOKEN_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp7/step10500/"
# CACHE_IMAGE_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp7/step10500/"

FOLDERS="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp9/step5500/global_step5500"
CACHE_TOKEN_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp9/step5500/"
CACHE_IMAGE_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp9/step5500/"

# FOLDERS="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/lzm_debug_step18000/global_step18000"
# CACHE_TOKEN_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/lzm_debug_step18000/"
# CACHE_IMAGE_FOLDER="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/lzm_debug_step18000/"


# 基础参数（使用默认值）
IMAGE_TOKENIZER_DIR="/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"
VAE_DIR="/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
# SOURCE_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage2_5e-5_force324/step74500/global_step74500/converted/"
SOURCE_DIR="/mmu_mllm_hdd_2/zhouyang12/models/onebase_1231_2wtoken_understanding"

# 可能需要调整的参数
DCP_CKPT_DIR="/mmu_mllm_hdd_2/zhouyang12/output/MuseV2/sana_v2/multi_scale/exp3.22/"
DCP_TAG="global_step100000"
MODEL_DIR="/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/"

# 可选参数
IMAGE_SIZE=512
DATASET="GenEval"
MAX_CONDITION_LENGTH=324
NUM_GENERATION_IMAGES=""

# 运行命令
PYTHONPATH=. python3 "$PYTHON_SCRIPT" \
  --folders "$FOLDERS" \
  --cache_token_folder "$CACHE_TOKEN_FOLDER" \
  --cache_image_folder "$CACHE_IMAGE_FOLDER" \
  --image_tokenizer_dir "$IMAGE_TOKENIZER_DIR" \
  --vae_dir "$VAE_DIR" \
  --source_dir "$SOURCE_DIR" \
  --dcp_ckpt_dir "$DCP_CKPT_DIR" \
  --dcp_tag "$DCP_TAG" \
  --model_dir "$MODEL_DIR" \
  --image_size "$IMAGE_SIZE" \
  --dataset "$DATASET" \
  --max_condition_length "$MAX_CONDITION_LENGTH"