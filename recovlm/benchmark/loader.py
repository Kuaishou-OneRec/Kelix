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

  def load(self, path: str) -> str:
    if not path.endswith(".txt"):
      path += ".txt"
    with open(os.path.join(self.library_dir, path), encoding="utf-8") as f:
      return f.read()
