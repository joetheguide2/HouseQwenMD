import os
import gc
import re
import random
import ast
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
SAMPLE_SIZE = None                         # set to None to use all rows
RANDOM_SEED = 42
SYSTEM_PROMPT = "You are a medical diagnostic expert who specializes in rare diseases."
MAX_SEQ_LENGTH = 8192
LORA_RANK = 64
BATCH_SIZE = 1                              # number of cases processed in parallel

# Generation parameters (identical to your interactive script)
GEN_KWARGS = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
    # pad_token_id is now handled via tokenizer.pad_token
}

# Output directory
OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths for continuous saving
BASE_CSV = os.path.join(OUTPUT_DIR, "base_results.csv")
FT_CSV   = os.path.join(OUTPUT_DIR, "ft_results.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
PLOT_ACCURACY = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
PLOT_TAGS = os.path.join(OUTPUT_DIR, "tag_presence.png")

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------- Load and sample data --------------------
df = pd.read_csv(CSV_PATH)
assert 'CaseSummary' in df.columns and 'Disease' in df.columns, \
    "CSV must contain 'CaseSummary' and 'Disease' columns"

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
print(f"Using {len(df)} cases for evaluation.")

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

def load_results_from_csv(csv_path):
    """
    Read a CSV saved by evaluate_model and convert it back to a list of result dictionaries.
    Handles the 'synonyms' column which was stored as a string representation of a list.
    """
    df_csv = pd.read_csv(csv_path)
    results = []
    for _, row in df_csv.iterrows():
        # Convert the string representation of list back to a Python list
        try:
            synonyms = ast.literal_eval(row['synonyms']) if pd.notna(row['synonyms']) else []
        except (SyntaxError, ValueError):
            synonyms = []   # fallback
        results.append({
            'case_idx': row['case_idx'],
            'true_disease': row['true_disease'],
            'synonyms': synonyms,
            'response': row['response'],
            'correct': bool(row['correct']),
            'has_think': bool(row['has_think']),
            'has_diagnose': bool(row['has_diagnose']),
        })
    return results

# Batched evaluation function with continuous saving and resume support
def evaluate_model(model, tokenizer, df, model_name, batch_size=BATCH_SIZE,
                   output_csv=None, overwrite=True):
    """
    Run batched evaluation on the dataframe.
    If output_csv is provided, results are appended to that CSV file after each batch.
    - overwrite=True:  if output_csv exists, it is deleted at the start.
    - overwrite=False: existing file is kept and new results are appended.
    Returns the full list of results for the processed data.
    """
    results = []

    # Handle existing CSV file according to overwrite flag
    if output_csv:
        if overwrite and os.path.exists(output_csv):
            os.remove(output_csv)
            print(f"  (Existing {output_csv} removed, starting fresh)")
        # If not overwriting, we keep the file and will append later.

    # Prepare tokenizer for batched generation
    tokenizer.padding_side = "left"          # important for decoder‑only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc=f"Evaluating {model_name}"):
        batch_df = df.iloc[i:i+batch_size]
        prompts = []
        indices = []

        # Build prompts for the batch
        for idx, row in batch_df.iterrows():
            case = row['CaseSummary']
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Look at the patient case and diagnose them. Case Summary : {case}"}
            ]
            # Get the full prompt string (without tokenizing yet)
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(prompt)
            indices.append(idx)

        # Tokenize the batch with padding and truncation
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GEN_KWARGS
            )

        # Decode only the newly generated tokens (skip the input prompt)
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Store results for each case in the batch
        batch_results = []
        for j, idx in enumerate(indices):
            row = df.loc[idx]
            true_disease = row['Disease']
            synonyms = parse_synonyms(row.get('Synonyms', ''))
            response = responses[j].strip()
            correct = disease_in_response(response, true_disease, synonyms)
            has_think = "</think>" in response
            has_diagnose = "<diagnose>" in response or "<diagnosis>" in response   # fallback
            res = {
                'case_idx': idx,
                'true_disease': true_disease,
                'synonyms': synonyms,
                'response': response,
                'correct': correct,
                'has_think': has_think,
                'has_diagnose': has_diagnose,
            }
            batch_results.append(res)
            results.append(res)

        # --- Continuous saving: append this batch to CSV ---
        if output_csv:
            batch_df = pd.DataFrame(batch_results)
            # Write header only if file does not exist yet (first batch ever)
            header = not os.path.exists(output_csv)
            batch_df.to_csv(output_csv, mode='a', header=header, index=False)

    return results

# -------------------- Base Model Evaluation (with resume logic) --------------------
base_results = None
base_model_loaded = False

if os.path.exists(BASE_CSV):
    existing_df = pd.read_csv(BASE_CSV)
    if len(existing_df) == len(df):
        print("\n" + "="*50)
        print("Base model results already complete. Loading from CSV.")
        base_results = load_results_from_csv(BASE_CSV)
    else:
        print("\n" + "="*50)
        print(f"Base model results partial ({len(existing_df)}/{len(df)}). Evaluating remaining cases.")
        processed_indices = set(existing_df['case_idx'])
        remaining_df = df[~df.index.isin(processed_indices)]

        # Load base model
        print("Loading base model (unsloth/Qwen2.5-1.5B-Instruct) in 4‑bit...")
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            fast_inference=True,
        )
        base_model = FastLanguageModel.for_inference(base_model)
        base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen2.5")
        base_model_loaded = True

        # Evaluate remaining cases (append to existing CSV)
        _ = evaluate_model(base_model, base_tokenizer, remaining_df,
                           "Base Model (resume)", output_csv=BASE_CSV, overwrite=False)

        # After completion, load full results from CSV
        base_results = load_results_from_csv(BASE_CSV)
else:
    print("\n" + "="*50)
    print("No base results found. Evaluating full set.")
    print("Loading base model (unsloth/Qwen2.5-1.5B-Instruct) in 4‑bit...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
    )
    base_model = FastLanguageModel.for_inference(base_model)
    base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen2.5")
    base_model_loaded = True

    base_results = evaluate_model(base_model, base_tokenizer, df,
                                  "Base Model", output_csv=BASE_CSV, overwrite=True)

# Clean up base model if it was loaded
if base_model_loaded:
    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# -------------------- Fine‑tuned Model Evaluation (with resume logic) --------------------
ft_results = None
ft_model_loaded = False

if os.path.exists(FT_CSV):
    existing_df = pd.read_csv(FT_CSV)
    if len(existing_df) == len(df):
        print("\n" + "="*50)
        print("Fine‑tuned model results already complete. Loading from CSV.")
        ft_results = load_results_from_csv(FT_CSV)
    else:
        print("\n" + "="*50)
        print(f"Fine‑tuned model results partial ({len(existing_df)}/{len(df)}). Evaluating remaining cases.")
        processed_indices = set(existing_df['case_idx'])
        remaining_df = df[~df.index.isin(processed_indices)]

        # Load fine‑tuned model
        print("Loading fine‑tuned model (./finetuned_lora) in 4‑bit...")
        ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
            model_name="./finetuned_lora",
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=LORA_RANK,
        )
        ft_model = FastLanguageModel.for_inference(ft_model)
        ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
        ft_model_loaded = True

        # Evaluate remaining cases (append to existing CSV)
        _ = evaluate_model(ft_model, ft_tokenizer, remaining_df,
                           "Fine‑tuned Model (resume)", output_csv=FT_CSV, overwrite=False)

        # After completion, load full results from CSV
        ft_results = load_results_from_csv(FT_CSV)
else:
    print("\n" + "="*50)
    print("No fine‑tuned results found. Evaluating full set.")
    print("Loading fine‑tuned model (./finetuned_lora) in 4‑bit...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name="./finetuned_lora",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
    )
    ft_model = FastLanguageModel.for_inference(ft_model)
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
    ft_model_loaded = True

    ft_results = evaluate_model(ft_model, ft_tokenizer, df,
                                "Fine‑tuned Model", output_csv=FT_CSV, overwrite=True)

# Clean up fine‑tuned model if it was loaded
if ft_model_loaded:
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

# Save detailed results to CSV (final combined file)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDetailed results saved to {OUTPUT_CSV}")
print(f"(Per‑model raw results are also available in {BASE_CSV} and {FT_CSV})")

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
