import string
import emoji
from typing import Dict
from .filter import FilterBase

# special characters
MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits \
                        + string.whitespace

OTHER_SPECIAL_CHARACTERS = """
    　    ￼’“”–ー一▬…✦�­£​•€«»°·═
×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖
◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚
゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖
」﴾》
"""

EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(MAIN_SPECIAL_CHARACTERS + OTHER_SPECIAL_CHARACTERS)
SPECIAL_CHARACTERS.update(EMOJI)


class SpecialCharactersFilter(FilterBase):

    def __init__(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = 0.25,
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, src: Dict[str, any]) -> bool:
        text = self.extract_all_text(src)

        special_char_ratio = (
            len([c for c in text if c in SPECIAL_CHARACTERS]) /
            len(text) if len(text) != 0 else 0.0
        )

        return self.min_ratio <= special_char_ratio <= self.max_ratio
