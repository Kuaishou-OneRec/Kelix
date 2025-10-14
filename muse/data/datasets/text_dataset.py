from typing import Dict, Any, Union, List
from transformers import AutoTokenizer
from muse.data.datasets.base import DistributedDataset
from muse.data.prompts import PromptLoader

class Qwen3Dataset(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               system_prompt: str = "default",
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.tokenizer_path = kwargs.get("tokenizer_path", None)
    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

  def process(self,
              sample: Dict[str, Any]):
    assert "messages" in sample["json"], \
        f"sample must contain `messages` key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]

    transformed_messages = []
    valid_message = True
    for msg in messages:
        role = msg.get("role")
        content_list = msg.get("content")
        if role and isinstance(content_list, list) and content_list and content_list[0].get("type") == "text":
            text_content = content_list[0].get("text")
            if text_content is not None:
                transformed_messages.append({"role": role, "content": text_content})
            else:
                valid_message = False
                break
        else:
            valid_message = False
            break
    
    if not valid_message or not transformed_messages:
      print(f"Warning: Message format is incorrect. Data: {messages}")

    if not messages or messages[0]["role"] != "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    text = self.tokenizer.apply_chat_template(
      transformed_messages,
      tokenize=False,
      add_generation_prompt=True
    )

    return {
      "vllm_inputs": {
        "prompt": text,
      },
      "annotation": "",
      "image_path": "",
      "video_path": "",
      "metadata": sample["raw"].loc["metadata"],
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
    }

   