MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct-LLaVA-CC3M-Pretrain-595K-bsz16x8

python3 infer.py --model_dir=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --step=latest \
    --eval_file=./eval_imgs.txt \
    --output_file=./eval_imgs_out_1epoch.txt