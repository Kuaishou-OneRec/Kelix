cd /llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit
source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate
conda activate ulmevalkit2
cf=blip3o_sft_step800
config=config/${cf}.json

echo "config: $config"
cat $config

work_dir=$1
max_infer_items=300000 \
PYTHONPATH=. \
nohup torchrun \
    --nproc_per_node=8 \
    run_eval_only.py --config config/${cf}.json \
    --eval-id default \
    --work-dir ${work_dir} \
    > ${work_dir}/eval_${cf}.out 2>&1 &
