#!/usr/bin/env bash
set -e

# 最简版：直接跑 run_eval_only.py（用于你调试“卡住”问题）
# 需要改路径就改下面几个变量即可。

ULMEVAL_DIR="/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit"
WORK_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp16x/exp163_0116sftv1_1e-4lr_directly_sft_debug/global_step4800/inference/DPGBench/outputs/ulmeval/aggresults/"
CONFIG="config/dpg_blip3o_sft.json"
EVAL_ID="default"

mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/run_eval_only.log"

bash -c "source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate && \
conda activate ulmevalkit2 && \
cd $ULMEVAL_DIR && \
max_infer_items=300000 PYTHONPATH=. \
torchrun --nproc_per_node=8 run_eval_only.py --config $CONFIG --eval-id $EVAL_ID --work-dir $WORK_DIR \
> $LOG_FILE 2>&1"

echo "log: $LOG_FILE"
