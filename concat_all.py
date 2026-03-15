import pandas as pd
import json
import os

# ----------------------------------------------------------------------
# List of all CSV files (from the image)
# ----------------------------------------------------------------------
df = pd.read_parquet("./training.parquet")
import re
import json
import pandas as pd

# ----------------------------------------------------------------------
# Helper: rebuild json column from CaseSummary and CoT
# ----------------------------------------------------------------------
def build_messages(row):
    system_prompt = "You are a medical diagnostic expert."
    user_prompt = f"Look at the patient case summary and diagnose them. Case summary: {row['CaseSummary']}"
    assistant_content = row['CoT'] if pd.notna(row['CoT']) else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content}
    ]
    return json.dumps(messages, ensure_ascii=False)

# ----------------------------------------------------------------------
# Step 1: Regenerate json column using cleaned CaseSummary and CoT
# ----------------------------------------------------------------------
df['json'] = df.apply(build_messages, axis=1)
print("✅ Json column regenerated from cleaned CaseSummary and CoT.\n")

# ----------------------------------------------------------------------
# Helper: extract the case summary from a json string
# ----------------------------------------------------------------------
def extract_case_summary_from_json(json_str):
    try:
        messages = json.loads(json_str)
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # The user prompt has a fixed prefix; extract the actual summary part
                prefix = "Look at the patient case summary and diagnose them. Case summary: "
                if content.startswith(prefix):
                    return content[len(prefix):]
                return content
    except:
        return None
    return None

# ----------------------------------------------------------------------
# Helper: extract assistant's CoT from json
# ----------------------------------------------------------------------
def extract_cot_from_json(json_str):
    try:
        messages = json.loads(json_str)
        for msg in messages:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    except:
        return None
    return None

# ----------------------------------------------------------------------
# Helper: extract diagnosis content from CoT (between <diagnosis> tags)
# ----------------------------------------------------------------------
def extract_diagnosis_content(cot):
    if pd.isna(cot):
        return None
    match = re.search(r'<diagnosis>(.*?)</diagnosis>', cot, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

# ----------------------------------------------------------------------
# Helper: check if text contains disease or any synonym
# ----------------------------------------------------------------------
def contains_disease_or_synonym(text, disease, synonyms_str):
    if pd.isna(text):
        return False
    text_lower = text.lower()
    if pd.notna(disease) and disease.lower() in text_lower:
        return True
    if pd.notna(synonyms_str):
        synonyms = [syn.strip().lower() for syn in synonyms_str.split(',') if syn.strip()]
        for syn in synonyms:
            if syn in text_lower:
                return True
    return False

# ----------------------------------------------------------------------
# Extract fields from json
# ----------------------------------------------------------------------
df['case_summary_from_json'] = df['json'].apply(extract_case_summary_from_json)
df['cot_from_json'] = df['json'].apply(extract_cot_from_json)

# ----------------------------------------------------------------------
# 1. Leakage check: case summary in json contains disease/synonyms?
# ----------------------------------------------------------------------
def leakage_check(row):
    return contains_disease_or_synonym(
        row['case_summary_from_json'],
        row.get('Disease'),
        row.get('Synonyms')
    )

df['leakage'] = df.apply(leakage_check, axis=1)
leakage_count = df['leakage'].sum()
total = len(df)
leakage_pct = (leakage_count / total) * 100 if total else 0

# ----------------------------------------------------------------------
# 2. Tag presence in CoT (from json)
# ----------------------------------------------------------------------
df['diag_content_from_json'] = df['cot_from_json'].apply(extract_diagnosis_content)
tag_present = df['diag_content_from_json'].notna().sum()
tag_pct = (tag_present / total) * 100 if total else 0

# ----------------------------------------------------------------------
# 3. Correctness among tagged (diagnosis content contains disease/synonyms)
# ----------------------------------------------------------------------
def correctness_check(row):
    if pd.isna(row['diag_content_from_json']):
        return False
    return contains_disease_or_synonym(
        row['diag_content_from_json'],
        row.get('Disease'),
        row.get('Synonyms')
    )

df['correct_tagged'] = df.apply(correctness_check, axis=1)
correct_among_tagged = df[df['diag_content_from_json'].notna()]['correct_tagged'].sum()
correct_tagged_pct = (correct_among_tagged / tag_present) * 100 if tag_present else 0

# ----------------------------------------------------------------------
# 4. Overall correct (tag present + content correct)
# ----------------------------------------------------------------------
overall_correct = df['correct_tagged'].sum()
overall_pct = (overall_correct / total) * 100 if total else 0

# ----------------------------------------------------------------------
# Print results
# ----------------------------------------------------------------------
print("=" * 70)
print("FINAL CHECKS ON JSON COLUMN")
print("=" * 70)

print(f"\n📌 Total rows: {total}")

print(f"\n🔍 LEAKAGE CHECK (case summary inside json contains disease/synonym)")
print(f"   Rows with leakage: {leakage_count} ({leakage_pct:.2f}%)")

print(f"\n🏷️  DIAGNOSIS TAG PRESENCE IN CoT (inside json)")
print(f"   Rows with <diagnosis> tags: {tag_present} ({tag_pct:.2f}%)")

print(f"\n✅ CORRECTNESS AMONG TAGGED (diagnosis content matches disease/synonym)")
print(f"   Correct among tagged: {correct_among_tagged} ({correct_tagged_pct:.2f}%)")

print(f"\n🎯 OVERALL CORRECT (tag present + content correct)")
print(f"   Overall correct: {overall_correct} ({overall_pct:.2f}%)")

# Optional: Show a few examples of leakage if any
if leakage_count > 0:
    print("\n" + "-" * 70)
    print("Examples of rows with leakage (case summary contains disease/synonym):")
    print("-" * 70)
    leaky_rows = df[df['leakage']].head(3)
    for idx, row in leaky_rows.iterrows():
        print(f"\nIndex {idx}:")
        print(f"Disease: {row['Disease']}")
        print(f"Synonyms: {row['Synonyms']}")
        print(f"Case summary snippet: {row['case_summary_from_json'][:200]}...")
print(df.info())
