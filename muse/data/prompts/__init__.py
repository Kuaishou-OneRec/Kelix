"""Prompt Loader"""
from typing import Optional
import os


class PromptLoader:
  """Load prompt content from file"""
  def __init__(self, library_dir: Optional[str] = None):
    if not library_dir:
      cur_dir = os.path.dirname(os.path.abspath(__file__))
      library_dir = os.path.join(cur_dir, "library")
    self.library_dir = library_dir

  def load(self, path_or_prompt: Optional[str] = None) -> str:
    if not path_or_prompt:
      return path_or_prompt
    raw = path_or_prompt
    if not path_or_prompt.endswith(".txt"):
      path_or_prompt += ".txt"
    if not os.path.exists(path_or_prompt):
      path_or_prompt = os.path.join(self.library_dir, path_or_prompt)
    if os.path.exists(path_or_prompt):
      with open(path_or_prompt, encoding="utf-8") as f:
        return f.read()
    return raw
