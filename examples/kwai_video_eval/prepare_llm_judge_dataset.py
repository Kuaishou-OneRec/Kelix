import json
import os
from pathlib import Path
from typing import List, Dict
import uuid
from recovlm.utils.data import get_media_info, save_parquet_dataset, \
    create_media_content
from recovlm.data.prompts import PromptLoader
import pyarrow.parquet as pq
import glob

def prepare_llm_judge_dataset(
    photo_dir: str,
    input_files: str,
    output_dir: str,
    prompt: str,
    system_prompt: str,
    num_shards: int = 1,
    tokenizer_path: str = None
):
    """
    Prepare LLM judge dataset from input files containing photo IDs and responses.
    
    Args:
        photo_dir: Directory containing photo information
        input_files: List of input JSON files containing photo IDs and responses
        output_dir: Directory to save the parquet dataset
        prompt: The prompt template to use for each sample
        system_prompt: The system prompt to use
        num_shards: Number of shards to split the dataset into
        tokenizer_path: Path to the tokenizer
    """
    prompt_loader = PromptLoader()
    prompt = prompt_loader.load(prompt)
    system_prompt = prompt_loader.load(system_prompt)
    samples = []
    all_input_files = []
    for input_file in input_files.split(","):
        all_input_files.extend(glob.glob(input_file))

    for input_file in all_input_files:
        # with open(input_file, 'r') as f:
        #     data = json.load(f)
        data = pq.read_table(input_file).to_pylist()
            
        for item in data:
            photo_id = item['photo']
            comments = item['content_list']
            id_list = item['id_list']
            assert len(comments) == len(id_list), \
                f"comments and id_list have different length: {len(comments)} != {len(id_list)}"
            # Get media info
            media_info = get_media_info(photo_id, photo_dir)
            if not media_info:
                print(f"Warning: Could not find media info for photo {photo_id}")
                continue
            media_content, videos_json, images_json = \
                create_media_content(media_info['media_path'])
            for comment, id in zip(comments, id_list):
                # Create messages in chat format
                message = {
                    "role": "user",
                    "content":
                        [{"type": "text", "text": "视频："}] + \
                        media_content + \
                        [{"type": "text", "text": "评论：" + comment}]
                }
                if prompt:
                    message["content"].append({"type": "text", "text": prompt})
                # Create sample
                sample = {
                    "images": images_json,
                    "videos": videos_json,
                    "source": "llm_judge",
                    "messages": json.dumps([message]),
                    "segments": "[]",  # No segments in this dataset
                    "metadata": json.dumps({
                        "photo_id": photo_id,
                        "comment_id": id
                    }),
                    "uuid": str(uuid.uuid4())
                }
                samples.append(sample)

    # Save to parquet dataset
    save_parquet_dataset(
        samples=samples,
        output_dir=output_dir,
        num_shards=num_shards,
        tokenizer_path=tokenizer_path,
        name="llm_judge_dataset"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare LLM judge dataset")
    parser.add_argument("--photo-dir", type=str, default=None, help="Directory containing photo information")
    parser.add_argument("--input-files", type=str, default=None, help="Input JSON files containing photo IDs and responses")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the parquet dataset")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt template to use for each sample")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt to use")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to split the dataset into")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to the tokenizer")
    
    args = parser.parse_args()
    
    prepare_llm_judge_dataset(
        photo_dir=args.photo_dir,
        input_files=args.input_files,
        output_dir=args.output_dir,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        num_shards=args.num_shards,
        tokenizer_path=args.tokenizer_path
    )
