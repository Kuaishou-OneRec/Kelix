import argparse
import torch

import os
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory of the pretrained model.")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="The directory of trained model.")
    
    parser.add_argument("--step", type=str, default=None,
                        help="The step to infer.")

    parser.add_argument("--eval_file", type=str, default=None,
                        help="The evaluation file")
    
    parser.add_argument("--output_file", type=str, default=None,
                        help="The prediction output file")

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_dir)

    model_config = Qwen2VLForConditionalGeneration.config_class.from_pretrained(args.model_dir)
    model = Qwen2VLForConditionalGeneration(model_config)
    
    step = args.step
    if step == "latest":
        with open(os.path.join(args.output_dir, "latest")) as f:
            step = f.read()
    else:
        step = f"global_step{step}"
    state_dict = torch.load(os.path.join(args.output_dir, step, "bf16", "pytorch_model.bin"))

    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()

    with open(args.output_file, "w", encoding="utf-8") as fout:
        with open(args.eval_file) as f:
            for line in f:
                img = line.strip()
                messages = messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img,
                                "resized_height": 224,
                                "resized_width": 224,
                            },
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                fout.write(f"Img:\n{img}\nCaption:\n{output_text}\n\n")
                fout.write("#" * 100)

if __name__ == '__main__':
    main()