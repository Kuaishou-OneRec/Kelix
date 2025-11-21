from typing import Dict, Any, Union, List
from transformers import AutoTokenizer
from muse.data.datasets.base import DistributedDataset
from muse.data.prompts import PromptLoader

class ChatMLDataset(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               system_prompt: str = "default",
               use_system_prompt: bool = True,
               add_prompt_loss: bool = True,
               tokenizer_path: str = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.use_system_prompt = use_system_prompt
    self.tokenizer_path = tokenizer_path
    if self.tokenizer_path:
      self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
    else:
      self.tokenizer = None

  def process(self,
              sample: Dict[str, Any]):
    assert "row" in sample, \
      f"sample must contain `row` key, but got {sample.keys()}"

    row = sample["row"]
    messages = json.loads(row.get("messages", "[]"))

    assert isinstance(messages, list) and len(messages) > 0, \
      f"messages must be a non-empty list, but got {messages}"

    if self.use_system_prompt and not messages[0]["role"] == "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)
    
    input_ids = []
    loss_mask = []

    for turn in messages:
      if turn["role"] == "system":
        _input_ids = self.tokenizer.apply_chat_template(turn, add_generation_prompt=False)
        _loss_mask = [int(self.add_prompt_loss)] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      elif turn["role"] == "user":
        _input_ids = self.tokenizer.apply_chat_template(turn, add_generation_prompt=True)
        _loss_mask = [int(self.add_prompt_loss)] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      elif turn["role"] == "assistant":
        _input_ids = self.tokenizer.apply_chat_template(turn)
        _loss_mask = [1] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      else:
        raise ValueError(f"Invalid role: {turn['role']}")

    # batchify
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    loss_mask = torch.tensor(loss_mask).unsqueeze(0)

    return {
      "input_ids": input_ids,
      "loss_mask": loss_mask,
    }

   