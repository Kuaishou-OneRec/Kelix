export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

MODEL_DIR=/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/
#MODEL_DIR=/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step10000/muse_converted/
DATASET_CONFIG=examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json
MODEL_DIR=${MODEL_DIR} DATASET_CONFIG=${DATASET_CONFIG}  bash examples/sana/ar_dit/inference/mpi_infer_custom_v2.sh
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

