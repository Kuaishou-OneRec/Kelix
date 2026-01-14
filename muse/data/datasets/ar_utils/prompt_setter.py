import torch

task2prompt_baseline = {
        "__default__": "You are a helpful assistant."
      }
task2prompt_coarse = {
      "image_edit": "You are a helpful assistant that can edit images.",
      "image_generation": "You are a helpful assistant in image generation.",
      "__default__": "You are a helpful assistant."
    }
task2prompt_fine = {
    "image_edit": "You are a helpful assistant that can edit images.",
    "image_generation": "You are a helpful assistant in image generation.",
    "ocr": "You are a helpful assistant that can recognize text in images.",
    "math": "You are a helpful assistant that can solve math problems.",
    "ai2d": "You are an assistant for AI2D, a task focused on scientific diagram structure parsing and semantic Q&A.",
    "chart": "You are a helpful assistant that can parse charts.",
    "code": "You are a helpful assistant that can recognize and understand code in images.",
    "__default__": "You are a helpful assistant."
}
class SystemPromptByTaks:
  def __init__(self, task2prompt=None):
    if task2prompt is None:
      task2prompt = {
        "__default__": "You are a helpful assistant."
      }
    self.task2prompt = task2prompt
    self.source_to_task = {
    }

  def classify_source2task(self, source):
    source = source.lower()
    if ("edit" in source and 'gen_' in source) or 'Gen_OmniGen2_X2I2'.lower() in source \
      or 'EditMetaQuery'.lower() in source or 'AnyEdit'.lower() in source \
        or 'Pico-banana'.lower() in source or 'Gen_PIPE'.lower() in source:
      return "image_edit"
    if "gen_" in source:
      return "image_generation"
    if "ocr" in source or 'write' in source or 'latex' in source:
      return "ocr"
    if "math" in source:
      return "math"
    if "ai2d" in source:
      return "ai2d"
    if "chart" in source:
      return "chart"
    if "code" in source or "sql" in source:
      return "code"
    return "others"


  def __call__(self, messages, source):
    if source not in self.source_to_task:
      task = self.classify_source2task(source)
      self.source_to_task[source] = task
      if torch.distributed.get_rank() == 0:
        print(f"find new source. task: {task}, source: {source}")
    else:
      task = self.source_to_task[source]
    sys_prompt = self.task2prompt.get(task, self.task2prompt["__default__"])
    if isinstance(sys_prompt, list):
      sys_prompt = random.choice(sys_prompt)
    if messages[0]["role"] == "system": return messages
    messages = [{"role": "system", "content": sys_prompt}] + messages
    # print("messages: ", messages)
    return messages