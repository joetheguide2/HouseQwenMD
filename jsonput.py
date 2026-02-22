import pandas as pd
import json

# --- Configuration ---
SYSTEM_PROMPT = "You are a medical diagnostic expert who specializes in rare diseases."
INPUT_FILE1 = "train1.csv"
INPUT_FILE2 = "train3.csv"
OUTPUT_FILE = "merged_final.csv"

# --- Process train1.csv ---
df1 = pd.read_csv(INPUT_FILE1)

def create_messages_from_row(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Look at the patient case and diagnose him. Case summary : {row['CaseSummary']}"},
        {"role": "assistant", "content": row['CoT']}
    ]
    return json.dumps(messages, ensure_ascii=False)

df1['messages'] = df1.apply(create_messages_from_row, axis=1)
df1 = df1[['messages']]  # keep only the messages column

# --- Process train2.csv ---
df2 = pd.read_csv(INPUT_FILE2)


def add_system_prompt(messages_json):
    try:
        messages = json.loads(messages_json)
        # Prepend system message (assumes no system message already present)
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return json.dumps(messages, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, IndexError):
        # If parsing fails, return original or handle appropriately
        # For simplicity, we'll return None and drop later, but here we keep original
        return messages_json

df2['messages'] = df2['messages'].apply(add_system_prompt)
df2 = df2[['messages']]  # keep only messages column

# --- Merge and shuffle ---
combined = pd.concat([df1, df2], ignore_index=True)
shuffled = combined.sample(frac=1).reset_index(drop=True)

# --- Save to CSV ---
shuffled.to_csv(OUTPUT_FILE, index=False)
print(len(combined))
print(f"Done. Output saved to {OUTPUT_FILE}")
