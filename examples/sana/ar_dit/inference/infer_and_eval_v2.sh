
MODEL_DIR=/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/
MODEL_DIR=${MODEL_DIR} bash examples/sana/ar_dit/inference/mpi_infer_custom.sh
work_dir=${MODEL_DIR}/inference/GenEval/outputs/ulmeval/aggresults/
source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate 
conda activate ulmevalkit2
cd /llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit
cf=blip3o_sft_step800
max_infer_items=300000 PYTHONPATH=. \
torchrun \
--nproc_per_node=8 \
run_eval_only.py \
--config config/${cf}.json \
--eval-id default \
--work-dir $work_dir \
> ${work_dir}/eval_${cf}.out 2>&1

