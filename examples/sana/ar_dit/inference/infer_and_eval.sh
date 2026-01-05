DCP_CKPT_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/ \
DCP_TAG=global_step52000 \
bash examples/sana/ar_dit/inference/mpi_infer_custom.sh
work_dir=${DCP_CKPT_DIR}/${DCP_TAG}/inference/GenEval/outputs/ulmeval/aggresults/
source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate 
conda activate ulmevalkit2
cd /llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit
max_infer_items=300000 PYTHONPATH=. \
torchrun \
--nproc_per_node=8 \
run_eval_only.py \
--config config/blip3o_sft_step800.json \
--eval-id default \
--work-dir $work_dir \
> ${work_dir}/eval_${cf}.out 2>&1

