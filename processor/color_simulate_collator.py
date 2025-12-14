import re
import cv2
from matplotlib import pyplot as plt
import torch
from transformers import AutoProcessor
from transformers import Qwen2VLProcessor,Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
import math
from PIL import Image
import numpy as np
from utils.cvdObserver import cvdSimulateNet
from network import CVDSimulateNetMachado
class ColorSimulateCollator:
    def __init__(self, processor, cvd_type="protan_80"):
        self.processor = processor
        self.patch_size = 16
        self.merge_size = 2
        self.cvdSimulateNet = cvdSimulateNet(cvd_type=cvd_type, cuda=False, batched_input=False)
        # if cvd_type.startswith("deutan"):
        #     severity = float(cvd_type.split("_")[-1])/100.
        #     self.cvdSimulateNet = CVDSimulateNetMachado(cvd_type="Deuteranomaly",severity=severity)
        # elif cvd_type.startswith("protan"):
        #     severity = float(cvd_type.split("_")[-1])/100.
        #     self.cvdSimulateNet = CVDSimulateNetMachado(cvd_type="Protanomaly",severity=severity)
    
    def __call__(self, inputs):
        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True) for example in inputs]
        # 读取图像
        image_inputs = []
        image_inputs_tmp = []

        for example in inputs:
            for message in example["messages"]:
                for content in message["content"]:
                    if content["type"] == "image":
                        image = cv2.imread(content["image"])
                        image_inputs_tmp.append(image)
        for image_input in image_inputs_tmp:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)/255.
            image_tensor = torch.from_numpy(image_input).permute(2, 0, 1).float()
            image_tensor = self.cvdSimulateNet(image_tensor)
            # normalize_factor = torch.tensor([0.586,0.293,0.025]).reshape(3, 1, 1).float().to(image_tensor.device)
            # image_tensor = image_tensor/(normalize_factor)
            image_tensor = image_tensor.clamp(0, 1).permute(1, 2, 0)
            image_np = image_tensor.numpy() # 不要转换精度，防止损失信息
            # image_pil = Image.fromarray(image_np)   
            image_inputs.append(image_np)
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, images_kwargs={"do_rescale": False})
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
        # # debug
        # plt.imshow(image_inputs[0])
        # plt.show()
        return batch
        
