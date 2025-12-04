# color_enhance_processor.py
import re
import cv2
import torch
from transformers import AutoProcessor
from transformers import Qwen2VLProcessor,Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
import math
from PIL import Image
import numpy as np
COLOR_WORDS = [
    "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "black",
    "white", "chocolate"
]

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

class ColorSensitiveCollator:
    def __init__(self, processor):
        self.processor = processor
        self.patch_size = 16
        self.merge_size = 2
        
    def __call__(self, inputs):
        # print(f"inputs{inputs[0]}") # debug
        # print("==================================")
        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False) for example in inputs]
        # 读取图像，智能resize，并padding至统一维度，便于处理. 统一用PIL格式，避免接口不明确
        image_inputs = []
        image_inputs_tmp = []
        max_width = 0
        max_height = 0
        for example in inputs:
            for message in example["messages"]:
                for content in message["content"]:
                    if content["type"] == "image" and content["image"]!=None:
                        image = cv2.imread(content["image"])
                        # 智能resize图像
                        h_bar, w_bar = smart_resize(image.shape[0], image.shape[1], factor=self.patch_size * self.merge_size)
                        image = cv2.resize(image, (w_bar, h_bar))
                        # 更新最大宽度和高度
                        max_width = max(max_width, w_bar)
                        max_height = max(max_height, h_bar)
                        image_inputs_tmp.append(image)
        for image_input in image_inputs_tmp:
            # padding至统一维度
            image_input_new = cv2.copyMakeBorder(image_input, 
                                                 0, max_height - image_input.shape[0], 0, max_width - image_input.shape[1], 
                                                 cv2.BORDER_CONSTANT, value=[128,128,128])
            # print(f"Image shape change from {image_input.shape} to {image_input_new.shape}. max_width={max_width}, max_height={max_height}")    # debug
            # 将图像从 BGR 转换为 RGB 格式
            image_rgb = cv2.cvtColor(image_input_new, cv2.COLOR_BGR2RGB)
            # 将 numpy 数组转换为 PIL 图像
            image_inputs.append(Image.fromarray(image_rgb))
        # Tokenize the texts and process the images
        # print(f"[DEBUG] imgtokens {len(texts)} {len(image_inputs)}")  # debug
        # print("[DEBUG] texts:")  # debug
        # print(f"[DEBUG] texts sample: {texts}")  # debug
        # for i, t in enumerate(texts):   # debug
            # count_img = t.count("<|image_pad|>")  # Qwen-VL 的图像占位符 # debug
            # print(f"  Sample {i}: {count_img} <image> tokens") # debug
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, images_kwargs={"do_rescale": False})
        image_batch = [torch.from_numpy(np.array(image)).permute(2, 0, 1)/255. for image in image_inputs]
        image_batch = torch.stack(image_batch, dim=0).to(batch["pixel_values"].device)
        batch["ori_images"] = image_batch
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2VLProcessor) or isinstance(self.processor, Qwen3VLProcessor):
            image_tokens = [151652,151653,151655]   
        else: 
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        # 后期可选功能
        # print(f"masks:{batch["attention_mask"].shape}-{batch["attention_mask"].sum()}")   # debug
        for i, single_input_ids in enumerate(batch["input_ids"]):
            tokens = self.processor.tokenizer.convert_ids_to_tokens(single_input_ids)
            # 生成颜色词相关的重要性掩码
            importance_mask = self._make_importance_mask(tokens)
            # 用重要性掩码替换attention_mask（转为float张量）
            batch["attention_mask"][i] = torch.tensor(importance_mask, dtype=batch["attention_mask"].dtype, device=batch["attention_mask"].device)
        # print(f"New masks:{batch["attention_mask"].shape}-{batch["attention_mask"].sum()}")   # debug
        return batch

    def _make_importance_mask(self, tokens, window=5):
        """
        对包含颜色词及其前后window个token的位置赋予较高权重（1），其余为0。
        匹配时忽略大小写，并排除非纯色词的前缀（如bred、greenhouse）。
        """
        n = len(tokens)
        # 初始化掩码为0（基础权重）
        mask = [0] * n

        for i, tok in enumerate(tokens):
            # 去掉token中的特殊符号前缀（如Ġ、##等，常见于BPE分词）
            clean_tok = re.sub(r"^[#Ġ▁]+", "", tok)
            clean_tok = clean_tok.lower()
            # 严格匹配颜色词
            if clean_tok in COLOR_WORDS:
                # 颜色词前后window范围内的token权重提升（设为1）
                start = max(0, i - window)
                end = min(n, i + window + 1)  # 注意end是开区间，所以+1
                for j in range(start, end):
                    mask[j] = 1
        if sum(mask)==0:
            mask[i] = 1
        return mask

        