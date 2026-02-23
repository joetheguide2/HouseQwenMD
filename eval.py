import os
import gc
import re
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# -------------------- Configuration --------------------
CSV_PATH = "Evaluation.csv"               # path to your CSV file
SAMPLE_SIZE = 1000
RANDOM_SEED = 42
SYSTEM_PROMPT = "You are a medical diagnostic expert who specializes in rare diseases."
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64

# Generation parameters (identical to your interactive script)
GEN_KWARGS = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
    "pad_token_id": None,   # will be set after tokenizer is loaded
}

OUTPUT_CSV = "evaluation_results.csv"
PLOT_ACCURACY = "accuracy_comparison.png"
PLOT_TAGS = "tag_presence.png"

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------- Load and sample data --------------------
df = pd.read_csv(CSV_PATH)
# Ensure required columns exist
assert 'CaseSummary' in df.columns and 'Disease' in df.columns, "CSV must contain 'CaseSummary' and 'Disease' columns"
# Sample 1000 rows
df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Sampled {len(df_sample)} cases for evaluation.")

# -------------------- Helper functions --------------------
def parse_synonyms(syn_str):
    """Convert a string of synonyms (comma separated) into a list of stripped strings."""
    if pd.isna(syn_str) or not isinstance(syn_str, str):
        return []
    return [s.strip() for s in syn_str.split(',') if s.strip()]

def disease_in_response(response, disease, synonyms):
    """Check if disease or any synonym appears in the response (case‑insensitive substring)."""
    response_lower = response.lower()
    if disease and disease.lower() in response_lower:
        return True
    for syn in synonyms:
        if syn.lower() in response_lower:
            return True
    return False

def get_response(model, tokenizer, case_summary):
    """Generate a diagnosis for a single case summary."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Look at the patient case and diagnose them. Case Summary : {case_summary}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_KWARGS)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    return response

def evaluate_model(model, tokenizer, df, model_name):
    """Run evaluation on the sampled dataframe and return list of result dicts."""
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model_name}"):
        case = row['CaseSummary']
        true_disease = row['Disease']
        synonyms = parse_synonyms(row.get('Synonyms', ''))  # column might be missing
        try:
            response = get_response(model, tokenizer, case)
        except Exception as e:
            print(f"Error on case {idx}: {e}")
            response = ""
        correct = disease_in_response(response, true_disease, synonyms)
        # For fine‑tuned model, check for tags
        has_think = "</think>" in response
        has_diagnose = "<diagnose>" in response   # note: in your example they used <diagnose> without closing?
        # The original instruction said <diagnose> tag, but the example used <diagnosis>. We'll check both?
        # Actually the user wrote: "tag presence of </think> and <diagnose>". We'll check exactly those strings.
        # In the first two examples, the model output <diagnosis>... maybe it's a typo. We'll check both to be safe.
        if not has_diagnose:
            has_diagnose = "<diagnosis>" in response  # fallback
        results.append({
            'case_idx': idx,
            'true_disease': true_disease,
            'synonyms': synonyms,
            'response': response,
            'correct': correct,
            'has_think': has_think,
            'has_diagnose': has_diagnose,
        })
    return results

# -------------------- Load base model --------------------
print("\n" + "="*50)
print("Loading base model (unsloth/Qwen2.5-1.5B-Instruct)...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,          # use 4‑bit to save memory (optional, can set False)
    fast_inference=False,
)
base_model = FastLanguageModel.for_inference(base_model)
base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen2.5")
GEN_KWARGS["pad_token_id"] = base_tokenizer.eos_token_id
base_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate base model
base_results = evaluate_model(base_model, base_tokenizer, df_sample, "Base Model")

# Clear memory
del base_model, base_tokenizer
gc.collect()
torch.cuda.empty_cache()

# -------------------- Load fine‑tuned model --------------------
print("\n" + "="*50)
print("Loading fine‑tuned model (./finetuned_lora)...")
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_lora",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,         # as in your original script
    fast_inference=False,
    max_lora_rank=LORA_RANK,
)
ft_model = FastLanguageModel.for_inference(ft_model)
ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
GEN_KWARGS["pad_token_id"] = ft_tokenizer.eos_token_id
ft_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate fine‑tuned model
ft_results = evaluate_model(ft_model, ft_tokenizer, df_sample, "Fine‑tuned Model")

# Clear memory
del ft_model, ft_tokenizer
gc.collect()
torch.cuda.empty_cache()

# -------------------- Combine and analyze results --------------------
# Convert results to DataFrames
base_df = pd.DataFrame(base_results).add_prefix('base_')
ft_df = pd.DataFrame(ft_results).add_prefix('ft_')

# Merge on case_idx
results_df = pd.concat([base_df, ft_df.drop(columns=['ft_case_idx'])], axis=1)

# Compute overall metrics
base_acc = results_df['base_correct'].mean()
ft_acc = results_df['ft_correct'].mean()
ft_think_pct = results_df['ft_has_think'].mean() * 100
ft_diagnose_pct = results_df['ft_has_diagnose'].mean() * 100

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Base model accuracy:      {base_acc*100:.2f}%")
print(f"Fine‑tuned model accuracy: {ft_acc*100:.2f}%")
print(f"Fine‑tuned </think> tag:   {ft_think_pct:.2f}%")
print(f"Fine‑tuned <diagnose> tag: {ft_diagnose_pct:.2f}%")

# Save detailed results to CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDetailed results saved to {OUTPUT_CSV}")

# -------------------- Generate plots --------------------
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

# Bar plot for accuracy
plt.subplot(1, 2, 1)
bars = plt.bar(['Base Model', 'Fine‑tuned Model'], [base_acc, ft_acc], color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, acc in zip(bars, [base_acc, ft_acc]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc*100:.1f}%',
             ha='center', va='bottom')
plt.title('Diagnosis Accuracy')

# Bar plot for tag presence (fine‑tuned only)
plt.subplot(1, 2, 2)
tags = ['</think>', '<diagnose>']
tag_pcts = [ft_think_pct, ft_diagnose_pct]
bars = plt.bar(tags, tag_pcts, color=['#2ca02c', '#d62728'])
plt.ylabel('Presence (%)')
plt.ylim(0, 100)
for bar, pct in zip(bars, tag_pcts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{pct:.1f}%',
             ha='center', va='bottom')
plt.title('Tag Presence in Fine‑tuned Model')

plt.tight_layout()
plt.savefig(PLOT_ACCURACY, dpi=150)
print(f"Accuracy plot saved to {PLOT_ACCURACY}")

# Optionally, create a separate figure for tag presence only
plt.figure(figsize=(6, 5))
bars = plt.bar(tags, tag_pcts, color=['#2ca02c', '#d62728'])
plt.ylabel('Presence (%)')
plt.ylim(0, 100)
for bar, pct in zip(bars, tag_pcts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{pct:.1f}%',
             ha='center', va='bottom')
plt.title('Tag Presence in Fine‑tuned Model')
plt.tight_layout()
plt.savefig(PLOT_TAGS, dpi=150)
print(f"Tag presence plot saved to {PLOT_TAGS}")

print("\nEvaluation complete.")
