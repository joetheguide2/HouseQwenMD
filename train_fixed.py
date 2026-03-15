import unsloth
import numpy as np
import pandas as pd
import gc                    
import torch
import json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

max_seq_length = 4096
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    #unsloth/gemma-3-1b-it
    max_seq_length=max_seq_length,         
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
)

tokenizer.model_max_length = max_seq_length  

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

df = pd.read_parquet("./cleaned_train.parquet")
df['messages'] = df['json'].apply(json.loads)

tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

ds = df.copy()
ds["text"] = None
for i in range(len(ds)):
    messages = ds.loc[i, "messages"]
    ds.loc[i, "text"] = tokenizer.apply_chat_template(messages, tokenize=False)

ds = Dataset.from_pandas(ds[["text"]])

def check_lengths(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=max_seq_length)
    return len(tokens["input_ids"])

lengths = ds.map(lambda x: {"length": check_lengths(x)})
print(f"Max token length after forced truncation: {max(lengths['length'])}")


gc.collect()
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="./checkpoints", 
        max_seq_length=max_seq_length,      
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.01,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        packing = True,
        report_to="tensorboard",
        gradient_checkpointing= "unsloth",
        save_strategy="steps",
        save_steps= 50,
        save_total_limit=2,
        logging_dir= "./runs"
    ),
)

trainer.train()

# import matplotlib.pyplot as plt
#
# # Get the log history
# log_history = trainer.state.log_history
#
# # Filter entries that contain training loss
# train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
# steps = [entry['step'] for entry in log_history if 'loss' in entry]
#
# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(steps, train_losses, label='Training Loss')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.grid(True)
#
# plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
# plt.show()

model.save_pretrained("finetuned_lora")
tokenizer.save_pretrained("finetuned_lora")

gc.collect()
torch.cuda.empty_cache()
