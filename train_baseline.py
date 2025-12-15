# train.py
import torch

# from swift.llm import get_model_tokenizer, load_dataset
from datasets import load_dataset
from processor.color_simulate_collator import ColorSimulateCollator
from trainer.color_sensitive_trainer import ColorSensitiveTrainer
# from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForImageTextToText, AutoProcessor, DataCollatorForSeq2Seq
from network import CVDSimulateNetMachado, colorFilter, hierarchicalModel
from trl import (
    SFTConfig,
    SFTTrainer,
)
import torchvision.transforms as transforms
import cv2

# ===== 基本设置 =====
MODEL_NAME = "/root/autodl-tmp/model/"  # 或任何支持 swift 的多模态模型
DATA_PATH = "/root/color_150k.json"  # SFT 数据：包含 {"instruction": ..., "output": ...}
OUTPUT_DIR = "./output_color_sensitive"
CVD_TYPE = "deutan_80"

# ===== 加载模型与processor =====
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, dtype="auto", device_map="auto"
)
dtype = model.dtype
device = model.device
processor = AutoProcessor.from_pretrained(MODEL_NAME)
# processor = ColorSensitiveProcessor.from_pretrained(MODEL_NAME)
# processor.__dict__.update(base_processor.__dict__)
collator = ColorSimulateCollator(processor,cvd_type=CVD_TYPE)
print(model)    # 显示模型结构

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    logging_steps=10,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=500,
    eval_strategy='steps',
    eval_steps=500,
    save_total_limit=1
)

sample_message = [
{    
    "image": "/root/autodl-tmp/images/train2017/000000033471.jpg",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe the image. If you see nothing, just tell me it is blank. "},{"type": "image", "image": "/root/autodl-tmp/images/train2017/000000033471.jpg"}]
        },
        # {
        #     "role": "assistant",
        #     "content": [{"type": "text", "text": "The image is a street scene with a car and a person."}]
        # }
    ]
}
]
inputs = collator(sample_message)
for key in inputs.keys():
    inputs[key] = inputs[key].to("cuda")
# debug
print(inputs)
from transformers import TextStreamer
text_streamer = TextStreamer(processor, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 256,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# ===== 加载数据 =====#
dataset = load_dataset("json", data_files=[DATA_PATH])["train"]

# Split dataset into train and eval
dataset_train = dataset.select(range(int(0.8 * len(dataset))))
dataset_eval = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

def process_example(example:dict):   
    # 注意：原始格式是ms-swift风格 {"messages": [{"role": ..., "content": ..., }]， "images": [...]}，
    # 现在改为 {"messages": [{"role": ..., "content": [{"type": "text", "text": ...}, {"type": "image", "image": ...}]}]}
    message_tmp = []  # 初始化空的对话
    for i,item in enumerate(example["messages"]):
        mesage_temp_i = {}
        mesage_temp_i["role"] = item["role"]
        if item["role"]=="user" and "<image>" in item["content"]:
            mesage_temp_i["content"] = [{"type": "image", "image": example["image"]},{"type": "text", "text": item["content"] },]
        elif item["role"]=="user" and "<image>" not in item["content"]:
            mesage_temp_i["content"] = [{"type": "text", "text": item["content"] }]
        else:
            mesage_temp_i["content"] = [{"type": "text", "text": item["content"] }]
        message_tmp.append(mesage_temp_i)
    example.pop("messages")
    example["messages"] = message_tmp
    return example

# 注意：datasets会将字段自动补全到相同格式，比如"type": "text", "text": xxx, "image": None
# 所以不要用map，用迭代处理
dataset_train = [process_example(example) for example in dataset_train]
dataset_eval = [process_example(example) for example in dataset_eval]
print(len(dataset_train))
import json
print("Dataset_train[0]:\n",json.dumps(dataset_train[0], ensure_ascii=False, indent=4))
# ===== 启动训练 =====
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    processing_class=processor,
    data_collator=collator,
)
model.train()
print("Start training...")
# 只训练ViT部分
for param in model.parameters():
    param.requires_grad = False
for param in model.visual.parameters():
    param.requires_grad = True
# calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of trainable parameters: {trainable_params}/{total_params}, total {trainable_params/total_params*100}%")

trainer.train()