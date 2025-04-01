import torch
from vllm import LLM, SamplingParams

def load_vllm_engine(cfg):
    devices = torch.cuda.device_count()
    print(devices)
    limit_mm_per_prompt={"image": cfg.model.limit_mm_per_prompt, 
                        "video": cfg.model.limit_mm_per_prompt}
    quantization = cfg.model.get('quantization', None)
    if quantization:
        vllm_engine = LLM(model=cfg.model.path, 
                            tensor_parallel_size=cfg.model.tp, 
                            limit_mm_per_prompt=limit_mm_per_prompt, 
                            quantization=cfg.model.quantization,
                            )
    else:
        vllm_engine = LLM(model=cfg.model.path, 
                            tensor_parallel_size=cfg.model.tp, 
                            limit_mm_per_prompt=limit_mm_per_prompt,
                            max_num_seqs=64,
                        )

    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        repetition_penalty=cfg.sampling.repetition_penalty,
        max_tokens=cfg.sampling.max_tokens,
        stop_token_ids=[],
    )
    return vllm_engine, sampling_params

def infer(vllm_engine, inputs, sampling_params):
    outputs = vllm_engine.generate(inputs, sampling_params)
    return outputs

