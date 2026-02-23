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
import torch
import gc
from unsloth import FastLanguageModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# --- Your existing loading code ---
max_seq_length = 4096
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_lora",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=False,
    max_lora_rank=lora_rank,
)

model = FastModel.for_inference(model)
tokenizer.model_max_length = max_seq_length

# (Optional) Ensure we have a proper chat template.
# If your fine‑tuned model already has one, this is not strictly necessary.
# Here we apply the "llama-3" template – change if needed.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen2.5",
)

# --- System prompt for the medical diagnostic expert ---
SYSTEM_PROMPT = "You are a medical diagnostic expert who specializes in rare diseases."

# --- Generation parameters ---
GEN_KWARGS = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
    "pad_token_id": tokenizer.eos_token_id,   # prevent warning about missing pad token
}

print("Medical Diagnostic Expert (type 'exit' or 'quit' to stop)")
print("-" * 50)

while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Build the conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_input}
    ]

    # Tokenize with the chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GEN_KWARGS
        )

    # Decode only the newly generated tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"Assistant: {response}")

    # Optional: clear CUDA cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
