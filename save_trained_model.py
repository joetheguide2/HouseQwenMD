from unsloth import FastLanguageModel
import torch

# Load your fine‑tuned LoRA adapter (the base model is automatically inferred)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./finetuned_lora",      # your adapter folder
    max_seq_length = 2048,                 # adjust to your needs
    dtype = None,                           # auto‑detect
    load_in_4bit = True,                    # load base model in 4‑bit for merging
)

# Merge the adapter and save a 4‑bit quantized version
model.save_pretrained_merged(
    "trained_16_bit",                       # output directory
    tokenizer,
    save_method = "merged_16bit",             # creates a 4‑bit merged model
)
