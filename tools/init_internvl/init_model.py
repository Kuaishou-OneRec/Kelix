from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from internvl.configuration_internvl_chat import InternVLChatConfig
from internvl.modeling_internvl_chat import InternVLChatModel
from internvl.modeling_internlm2 import InternLM2ForCausalLM
import os
import argparse
import torch
import logging

# 特殊token常量定义 (从internvl.train.constants)
IMG_START_TOKEN    = "<image>"
IMG_END_TOKEN      = "</image>"
IMG_CONTEXT_TOKEN  = "<im_patch>"
QUAD_START_TOKEN   = "<quad>"
QUAD_END_TOKEN     = "</quad>"
REF_START_TOKEN    = "<ref>"
REF_END_TOKEN      = "</ref>"
BOX_START_TOKEN    = "<box>"
BOX_END_TOKEN      = "</box>"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_argument_parser():
    """
    创建并配置命令行参数解析器
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default='/llm_reco/chuchenglong/InternVL/models/Qwen/Qwen2.5-3B-Instruct',
                        help="预训练LLM的目录")

    parser.add_argument("--vision_encoder_dir", type=str, default='/llm_reco/chuchenglong/InternVL/models/OpenGVLab/InternViT-300M-448px-V2_5',
                        help="预训练视觉模型(ViT)的目录")

    parser.add_argument("--new_model_dir", type=str, default='/llm_reco/chuchenglong/InternVL/models/Megred_model/4B',
                        help="新合并模型的输出保存目录")
                        
    # InternVL配置参数
    parser.add_argument("--force_image_size", type=int, default=448,
                        help="设置图像大小")
    parser.add_argument("--down_sample_ratio", type=float, default=0.5,
                        help="下采样比率")
    parser.add_argument("--pad2square", action="store_true",
                        help="是否将图像填充为正方形")
    parser.add_argument("--conv_style", type=str, default="internlm2-chat",
                        help="对话模板样式")
    parser.add_argument("--vision_select_layer", type=int, default=-1,
                        help="选择使用的ViT层，-1表示最后一层")
    parser.add_argument("--dynamic_image_size", type=bool, default=True,
                        help="是否使用动态图像大小")
    parser.add_argument("--use_thumbnail",  type=bool, default=True,
                        help="是否使用缩略图")
    parser.add_argument("--ps_version", type=str, default="v2",
                        choices=["v1", "v2"],
                        help="像素混洗版本")
    parser.add_argument("--min_dynamic_patch", type=int, default=1,
                        help="最小动态patch数量")
    parser.add_argument("--max_dynamic_patch", type=int, default=12,
                        help="最大动态patch数量")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                        help="Drop path率")

    return parser

def main():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()
    
    # 1. 分别加载语言模型、视觉模型和tokenizer
    logger.info(f'加载语言模型: {args.model_dir}')
    llm_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # 确保词嵌入权重共享
    if hasattr(llm_config, 'tie_word_embeddings'):
        llm_config.tie_word_embeddings = True
    
    # 设置flash attention
    if hasattr(llm_config, 'model_type') and llm_config.model_type == 'internlm2':
        model_type = InternLM2ForCausalLM
        llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        logger.info('使用flash_attention_2 (InternLM模型)')
    else:
        model_type = AutoModelForCausalLM
        if hasattr(llm_config, '_attn_implementation'):
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA/other
        logger.info('使用flash_attention_2 (非InternLM模型)')
    
    text_model = model_type.from_pretrained(
        args.model_dir, 
        torch_dtype=torch.bfloat16,
        config=llm_config, 
        trust_remote_code=True
    )
    # 语言模型初始化后即可 tie embeddings，以保证 lm_head 与输入嵌入共享权重
    if hasattr(text_model, "tie_weights"):
        text_model.tie_weights()
    elif hasattr(text_model, "tie_word_embeddings"):
        text_model.tie_word_embeddings()
    logger.info("已对语言模型调用 tie_weights()")
    
    logger.info(f'加载视觉模型: {args.vision_encoder_dir}')
    vision_config = AutoConfig.from_pretrained(args.vision_encoder_dir, trust_remote_code=True)
    vision_config.drop_path_rate = args.drop_path_rate
    
    vision_model = AutoModel.from_pretrained(
        args.vision_encoder_dir, 
        torch_dtype=torch.bfloat16,
        config=vision_config,
        trust_remote_code=True
    )
    
    # 2. 加载tokenizer，并添加特殊token
    logger.info(f'加载Tokenizer: {args.model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, 
        add_eos_token=False,
        trust_remote_code=True
    )
    
    # 添加特殊token (与InternVLChatModel训练过程一致)
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    logger.info(f"添加了 {num_new_tokens} 个特殊token")
    
    # 获取IMG_CONTEXT_TOKEN的ID
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    # 3. 创建完整的InternVL配置
    logger.info('创建InternVLChatConfig...')
    # 4. 构造 InternVLChatConfig
    vc = vision_config.to_dict()
    lc = llm_config.to_dict()
    internvl_chat_config = InternVLChatConfig(
        vision_config=vc, llm_config=lc,
        downsample_ratio=args.down_sample_ratio,
        pad2square=args.pad2square,
        template=args.conv_style,
        select_layer=args.vision_select_layer,
        dynamic_image_size=args.dynamic_image_size,
        use_thumbnail=args.use_thumbnail,
        ps_version=args.ps_version,
        min_dynamic_patch=args.min_dynamic_patch,
        max_dynamic_patch=args.max_dynamic_patch
    )
    internvl_chat_config.force_image_size = args.force_image_size
    
    # 4. 创建多模态模型
    logger.info('创建InternVLChatModel...')
    model = InternVLChatModel(
        config=internvl_chat_config,
        vision_model=vision_model,
        language_model=text_model
    )
    model.config.llm_config = text_model.config
    model.config.vision_config = vision_model.config
    # 1) 设置 num_image_token
    patch_size = model.config.vision_config.patch_size
    model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
    print(f"设置 num_image_token = {model.num_image_token}")
    
    # 2) 调整 position embeddings（如果需要）
    if model.config.vision_config.image_size != args.force_image_size:
        print(f'调整position embedding从 {model.config.vision_config.image_size} 到 {args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=args.force_image_size,
            patch_size=patch_size
        )
        model.config.vision_config.image_size = args.force_image_size
    
    # 3) 关闭语言模型缓存
    model.language_model.config.use_cache = False
    
    
    # 5. 为特殊token调整模型词表大小
    if num_new_tokens > 0:
        # resize embeddings 
        model.language_model.resize_token_embeddings(len(tokenizer))
        # 平均初始化新 token 的嵌入
        emb_weight = model.language_model.get_output_embeddings().weight.data
        avg_emb = emb_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
        emb_weight[-num_new_tokens:] = avg_emb
        # 更新 vocab_size
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)


    
    # 6) 设置 img_context_token_id
    model.img_context_token_id = img_context_token_id
    
    sd = model.state_dict()
    print(sd)

    
    os.makedirs(args.new_model_dir, exist_ok=True)
    # 不传 state_dict，使用 HF 默认 save 会自动处理 tie 权重
    model.save_pretrained(args.new_model_dir)
    tokenizer.save_pretrained(args.new_model_dir)

    logger.info(f"完成，已保存到 {args.new_model_dir}")
    
    logger.info(f"模型和 tokenizer 已保存到: {args.new_model_dir}")
    logger.info(f"成功创建 InternVL 模型，添加了 {num_new_tokens} 个特殊 token")

if __name__ == "__main__":
    main()
