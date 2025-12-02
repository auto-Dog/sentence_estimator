# train.py
import torch

from swift.llm import get_model_tokenizer, load_dataset
from processor.color_sensitive_processor import ColorSensitiveProcessor
from trainer.color_sensitive_trainer import ColorSensitiveTrainer
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from network import CVDSimulateNetMachado, colorFilter
# ===== 基本设置 =====
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # 或任何支持 swift 的多模态模型
DATA_PATH = "your_dataset.json"  # SFT 数据：包含 {"instruction": ..., "output": ...}
OUTPUT_DIR = "./output_color_sensitive"
CVD_TYPE = "Deuteranomaly"
CVD_SEVERITY = 1.0

# ===== 加载模型与processor =====
processor = ColorSensitiveProcessor(MODEL_NAME)

model,tokenizer = get_model_tokenizer(MODEL_NAME, trust_remote_code=True)[0]

# Combine VLM model, training model and Degradation model
model = torch.nn.Sequential(
    CVDSimulateNetMachado(cvd_type=CVD_TYPE, severity=CVD_SEVERITY),
    colorFilter(),
    model,
)

training_args = Seq2SeqTrainingArguments(
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
)

# ===== 加载数据 =====#
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def collate_fn(batch):
    texts = [b["instruction"] + "\n" + b["output"] for b in batch]
    processed = [processor(t, return_tensors="pt") for t in texts]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [p["input_ids"].squeeze(0) for p in processed], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [p["attention_mask"].squeeze(0) for p in processed], batch_first=True, padding_value=0
    )
    importance_mask = torch.nn.utils.rnn.pad_sequence(
        [p["importance_mask"].squeeze(0) for p in processed], batch_first=True, padding_value=1.0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "importance_mask": importance_mask,
        "labels": input_ids.clone(),
    }

# ===== 启动训练 =====
trainer = ColorSensitiveTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()
