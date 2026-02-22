import unsloth
import numpy as np
import pandas as pd


df = pd.read_csv("./merged_shuffled.csv")

from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vllm fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 42,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen2.5",
)

import json

df['messages'] = df['messages'].apply(json.loads)
ds = df
ds["text"] = ds["messages"]
print(type(ds.loc[0, "messages"]))
for i in range(len(ds)):
    messages = ds["messages"][i]
    ds.loc[i, "text"] = tokenizer.apply_chat_template(messages, tokenize=False)

from datasets import Dataset
ds = Dataset.from_pandas(ds)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        max_seq_length= 4096,
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        warmup_ratio = 0.01,
        num_train_epochs = 2,
        learning_rate = 1e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


trainer.train()

model.save_pretrained("finetuned_lora")
tokenizer.save_pretrained("finetuned_lora")
