export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

DCP_CKPT_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp22_ar_dit_324tokens_1e-4_reproduce
DCP_TAG=global_step24000


# DCP_CKPT_DIR=${DCP_CKPT_DIR} DCP_TAG=${DCP_TAG}  bash examples/sana/ar_dit/inference/mpi_infer_custom.sh

#DCP_CKPT_DIR=${DCP_CKPT_DIR} DCP_TAG=${DCP_TAG}  bash examples/sana/ar_dit/inference/mpi_infer_custom.sh
work_dir=${DCP_CKPT_DIR}/${DCP_TAG}/inference/GenEval/outputs/ulmeval/aggresults/
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

