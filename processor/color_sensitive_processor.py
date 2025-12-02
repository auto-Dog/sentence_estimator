# color_sensitive_processor.py
import re
import torch
from transformers import AutoProcessor

COLOR_WORDS = [
    "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "black",
    "white", "chocolate"
]

class ColorSensitiveProcessor:
    def __init__(self, base_processor_name):
        self.processor = AutoProcessor.from_pretrained(base_processor_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def __call__(self, text, **kwargs):
        inputs = self.processor(text, **kwargs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        importance_mask = self._make_importance_mask(tokens)
        inputs["importance_mask"] = torch.tensor(importance_mask, dtype=torch.float)
        return inputs

    def _make_importance_mask(self, tokens, window=5):
        """
        对包含颜色词及其前后 window 个 token 的位置赋予较高权重（例如2.0），其余为1.0。
        匹配时忽略大小写，并排除非纯色词的前缀（如 bred、greenhouse）。
        """
        n = len(tokens)
        mask = [1.0] * n

        for i, tok in enumerate(tokens):
            # 去掉特殊符号前缀（Ġ, ##等）
            clean_tok = re.sub(r"^[#Ġ▁]+", "", tok.lower())

            # 严格匹配颜色词
            if clean_tok in COLOR_WORDS:
                start = max(0, i - window)
                end = min(n, i + window + 1)
                for j in range(start, end):
                    mask[j] = 2.0  # 权重翻倍

        return mask
