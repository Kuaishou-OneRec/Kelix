import torch
from torch.utils.data import DataLoader
from recovlm.data.datasets import ChatCompletionDataset

"""
    # dataset = LLaVA_CC3M_Dataset(
    #     source="/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/",
    #     processor_path="/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14"
    # )
    # for idx, batch in enumerate(DataLoader(dataset, batch_size=3, collate_fn=dataset.build_collate_fn())):
    #     print(idx, batch)
    #     #print(dataset.processor.tokenizer.decode(batch["input_ids"]))
    #     if idx >= 1:
    #         break
    # for key, tensor in batch.items():
    #     print(key, tensor.shape)
    # for input_ids in batch["input_ids"]:
    #     print("=" * 10)
    #     print(dataset.processor.tokenizer.decode(input_ids))
"""

TOKENIZER = "/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct"


def test_chat_completion():
  records = [
      {
          "conversations": [
              {"role": "user", "content": "Hello!"},
              {"role": "assistant", "content": "你好👋!"},
          ]
      },
      {
          "conversations": [
              {"role": "user", "content": "こんにちは!"},
              {"role": "assistant", "content": "你好!"},
          ]
      }
  ]
  dataset = ChatCompletionDataset(
      source=records,
      tokenizer=TOKENIZER,
      input_key="conversations",
      system_prompt="You are RecoVLM",
      chat_template="chat_template_with_generation_tag",
      max_length=128
  )

  ans = {
      'input_ids': torch.tensor(
          [
              [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
               198, 151644, 872, 198, 9707, 0, 151645, 198, 151644,
               77091, 198, 108386, 145707, 0, 151645, 198],
              [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
               198, 151644, 872, 198, 89015, 0, 151645, 198, 151644,
               77091, 198, 108386, 0, 151645, 198, 151643]]),
      'attention_mask': torch.tensor(
          [
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
          ]),
      'loss_mask': torch.tensor(
          [
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
          ])
  }

  for batch in DataLoader(dataset,
                          batch_size=2,
                          shuffle=False,
                          collate_fn=dataset.collate_fn):
    for key, t in batch.items():
      assert torch.allclose(t, ans[key])
