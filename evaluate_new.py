import os
import gc
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
CSV_PATH = "Evaluation.csv"
MAX_SAMPLES_PER_DISEASE = 2                # samples per disease (change as needed)
RANDOM_SEED = 42
SYSTEM_PROMPT = "You are a medical diagnostic expert who specializes in rare diseases."
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64

# Generation parameters
GEN_KWARGS = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
    "pad_token_id": None,   # will be set after tokenizer is loaded
}

# Output folder
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------- Load and sample data (stratified by disease) --------------------
df = pd.read_csv(CSV_PATH)
assert 'CaseSummary' in df.columns and 'Disease' in df.columns, "CSV must contain 'CaseSummary' and 'Disease' columns"

# Group by disease and sample up to MAX_SAMPLES_PER_DISEASE cases per disease
disease_groups = df.groupby('Disease')
sampled_indices = []
for disease, group in disease_groups:
    n = min(len(group), MAX_SAMPLES_PER_DISEASE)
    if n > 0:
        sampled = group.sample(n=n, random_state=RANDOM_SEED)
        sampled_indices.extend(sampled.index.tolist())

df_eval = df.loc[sampled_indices].reset_index(drop=True)
print(f"Sampled {len(df_eval)} cases from {df_eval['Disease'].nunique()} unique diseases.")
print(f"Each disease contributes up to {MAX_SAMPLES_PER_DISEASE} cases.")

# -------------------- Helper functions --------------------
def parse_synonyms(syn_str):
    if pd.isna(syn_str) or not isinstance(syn_str, str):
        return []
    return [s.strip() for s in syn_str.split(',') if s.strip()]

def disease_in_response(response, disease, synonyms):
    response_lower = response.lower()
    if disease and disease.lower() in response_lower:
        return True
    for syn in synonyms:
        if syn.lower() in response_lower:
            return True
    return False

def get_response(model, tokenizer, case_summary):
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
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model_name}"):
        case = row['CaseSummary']
        true_disease = row['Disease']
        synonyms = parse_synonyms(row.get('Synonyms', ''))
        try:
            response = get_response(model, tokenizer, case)
        except Exception as e:
            print(f"Error on case {idx}: {e}")
            response = ""
        correct = disease_in_response(response, true_disease, synonyms)
        has_think = "</think>" in response
        has_diagnose = "<diagnose>" in response or "<diagnosis>" in response
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
    load_in_4bit=False,
    fast_inference=False,
)
base_model = FastLanguageModel.for_inference(base_model)
base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen2.5")
GEN_KWARGS["pad_token_id"] = base_tokenizer.eos_token_id
base_model.to("cuda" if torch.cuda.is_available() else "cpu")

base_results = evaluate_model(base_model, base_tokenizer, df_eval, "Base Model")
del base_model, base_tokenizer
gc.collect()
torch.cuda.empty_cache()

# -------------------- Load fine‑tuned model --------------------
print("\n" + "="*50)
print("Loading fine‑tuned model (./finetuned_lora)...")
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_lora",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=LORA_RANK,
)
ft_model = FastLanguageModel.for_inference(ft_model)
ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
GEN_KWARGS["pad_token_id"] = ft_tokenizer.eos_token_id
ft_model.to("cuda" if torch.cuda.is_available() else "cpu")

ft_results = evaluate_model(ft_model, ft_tokenizer, df_eval, "Fine‑tuned Model")
del ft_model, ft_tokenizer
gc.collect()
torch.cuda.empty_cache()

# -------------------- Combine and compute per‑disease metrics --------------------
base_df = pd.DataFrame(base_results).add_prefix('base_')
ft_df = pd.DataFrame(ft_results).add_prefix('ft_')
results_df = pd.concat([base_df, ft_df.drop(columns=['ft_case_idx'])], axis=1)

# Overall metrics
total_cases = len(results_df)
base_acc = results_df['base_correct'].mean()
ft_acc = results_df['ft_correct'].mean()
ft_think_pct = results_df['ft_has_think'].mean() * 100
ft_diagnose_pct = results_df['ft_has_diagnose'].mean() * 100

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Total cases evaluated: {total_cases}")
print(f"Unique diseases: {results_df['base_true_disease'].nunique()}")
print(f"Base model accuracy:      {base_acc*100:.2f}%")
print(f"Fine‑tuned model accuracy: {ft_acc*100:.2f}%")
print(f"Fine‑tuned </think> tag:   {ft_think_pct:.2f}%")
print(f"Fine‑tuned <diagnose> tag: {ft_diagnose_pct:.2f}%")

# Per‑disease recall (accuracy per disease)
per_disease = results_df.groupby('base_true_disease').agg(
    cases=('base_correct', 'count'),
    base_correct=('base_correct', 'sum'),
    ft_correct=('ft_correct', 'sum')
).reset_index()
per_disease['base_recall'] = per_disease['base_correct'] / per_disease['cases']
per_disease['ft_recall'] = per_disease['ft_correct'] / per_disease['cases']
per_disease['diff'] = per_disease['ft_recall'] - per_disease['base_recall']
per_disease = per_disease.sort_values('base_recall', ascending=False)  # sort by base recall for plotting

# Macro‑average recall (unweighted)
macro_base = per_disease['base_recall'].mean()
macro_ft = per_disease['ft_recall'].mean()

# Weighted average recall (by number of cases per disease)
weighted_base = (per_disease['base_recall'] * per_disease['cases']).sum() / per_disease['cases'].sum()
weighted_ft = (per_disease['ft_recall'] * per_disease['cases']).sum() / per_disease['cases'].sum()

print(f"\nMacro‑average recall (unweighted): Base = {macro_base*100:.2f}%, FT = {macro_ft*100:.2f}%")
print(f"Weighted average recall (by cases): Base = {weighted_base*100:.2f}%, FT = {weighted_ft*100:.2f}%")

# Count diseases where each model is better / tie
better_base = (per_disease['base_recall'] > per_disease['ft_recall']).sum()
better_ft = (per_disease['ft_recall'] > per_disease['base_recall']).sum()
tie = (per_disease['base_recall'] == per_disease['ft_recall']).sum()
print(f"\nDiseases where Base > FT: {better_base}")
print(f"Diseases where FT > Base: {better_ft}")
print(f"Diseases where Tie:       {tie}")

# Save detailed results
results_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)
per_disease.to_csv(os.path.join(RESULTS_DIR, "per_disease_metrics.csv"), index=False)
print(f"\nDetailed results saved to {RESULTS_DIR}/")

# -------------------- Generate plots --------------------
sns.set_style("whitegrid")

# 1. Per‑disease recall comparison across all diseases (sorted by base recall)
plt.figure(figsize=(14, 6))
x = np.arange(len(per_disease))  # disease index
plt.plot(x, per_disease['base_recall'], 'o-', markersize=2, linewidth=0.5, label='Base Model', color='#1f77b4', alpha=0.7)
plt.plot(x, per_disease['ft_recall'], 'o-', markersize=2, linewidth=0.5, label='Fine‑tuned Model', color='#ff7f0e', alpha=0.7)
plt.xlabel('Disease Index (sorted by base recall)')
plt.ylabel('Recall (Accuracy per Disease)')
plt.title('Per‑Disease Recall for Both Models')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "per_disease_recall_lines.png"), dpi=150)
plt.close()

# 2. Histogram of recall differences
plt.figure(figsize=(8, 5))
plt.hist(per_disease['diff'].dropna(), bins=30, edgecolor='black', color='#2ecc71', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Difference in Recall (Fine‑tuned − Base)')
plt.ylabel('Number of Diseases')
plt.title('Distribution of Per‑Disease Recall Differences')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "recall_diff_histogram.png"), dpi=150)
plt.close()

# 3. Bar chart of better/worse counts
plt.figure(figsize=(6, 5))
categories = ['Base > FT', 'FT > Base', 'Tie']
counts = [better_base, better_ft, tie]
colors = ['#3498db', '#e67e22', '#7f8c8d']
bars = plt.bar(categories, counts, color=colors)
plt.ylabel('Number of Diseases')
plt.title('Comparison of Model Performance per Disease')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count),
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "better_worse_counts.png"), dpi=150)
plt.close()

# 4. Scatter plot of base vs ft recall (each point = a disease)
plt.figure(figsize=(6, 6))
plt.scatter(per_disease['base_recall'], per_disease['ft_recall'], alpha=0.5, s=10, c='blue')
plt.plot([0,1], [0,1], 'r--', linewidth=1, label='y=x')
plt.xlabel('Base Recall')
plt.ylabel('Fine‑tuned Recall')
plt.title('Per‑Disease Recall: Base vs Fine‑tuned')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "base_vs_ft_scatter.png"), dpi=150)
plt.close()

# 5. (Optional) For diseases with multiple cases, we could also plot a heatmap of correctness patterns
# But with 1000 diseases, this is sufficient.

# Also save a summary text file
with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("EVALUATION SUMMARY\n")
    f.write("="*50 + "\n")
    f.write(f"Total cases evaluated: {total_cases}\n")
    f.write(f"Unique diseases: {per_disease.shape[0]}\n")
    f.write(f"Base model accuracy: {base_acc*100:.2f}%\n")
    f.write(f"Fine‑tuned model accuracy: {ft_acc*100:.2f}%\n")
    f.write(f"Fine‑tuned </think> tag presence: {ft_think_pct:.2f}%\n")
    f.write(f"Fine‑tuned <diagnose> tag presence: {ft_diagnose_pct:.2f}%\n")
    f.write(f"\nMacro‑average recall (unweighted): Base = {macro_base*100:.2f}%, FT = {macro_ft*100:.2f}%\n")
    f.write(f"Weighted average recall (by cases): Base = {weighted_base*100:.2f}%, FT = {weighted_ft*100:.2f}%\n")
    f.write(f"\nDiseases where Base > FT: {better_base}\n")
    f.write(f"Diseases where FT > Base: {better_ft}\n")
    f.write(f"Diseases where Tie:       {tie}\n")

print(f"\nAll results and plots saved in '{RESULTS_DIR}/'")
print("Evaluation complete.")
