import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2  # optional, install with: pip install matplotlib-venn

# Load the evaluation results
df = pd.read_csv("evaluation_results.csv")

# Convert boolean columns (they are stored as strings? Actually CSV saves bools as True/False, pandas reads them as bool)
base_correct = df['base_correct']
ft_correct = df['ft_correct']

# Compute counts
total = len(df)
base_only = ((base_correct) & (~ft_correct)).sum()
ft_only = ((~base_correct) & (ft_correct)).sum()
both = (base_correct & ft_correct).sum()
neither = ((~base_correct) & (~ft_correct)).sum()

base_total = base_correct.sum()
ft_total = ft_correct.sum()

print("Correctness breakdown:")
print(f"Both models correct:        {both} ({both/total*100:.1f}%)")
print(f"Only base model correct:    {base_only} ({base_only/total*100:.1f}%)")
print(f"Only fine‑tuned correct:    {ft_only} ({ft_only/total*100:.1f}%)")
print(f"Neither model correct:      {neither} ({neither/total*100:.1f}%)")
print(f"Base model total correct:   {base_total} ({base_total/total*100:.1f}%)")
print(f"Fine‑tuned total correct:   {ft_total} ({ft_total/total*100:.1f}%)")

# -------------------- 1. Grouped bar chart --------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart for individual model performance
models = ['Base Model', 'Fine‑tuned Model']
correct_counts = [base_total, ft_total]
incorrect_counts = [total - base_total, total - ft_total]

x = np.arange(len(models))
width = 0.35

ax[0].bar(x - width/2, correct_counts, width, label='Correct', color='#2ecc71')
ax[0].bar(x + width/2, incorrect_counts, width, label='Incorrect', color='#e74c3c')
ax[0].set_xticks(x)
ax[0].set_xticklabels(models)
ax[0].set_ylabel('Number of cases')
ax[0].set_title('Correct vs Incorrect Predictions')
ax[0].legend()
# Add value labels
for i, (corr, incorr) in enumerate(zip(correct_counts, incorrect_counts)):
    ax[0].text(i - width/2, corr + 2, str(corr), ha='center', va='bottom')
    ax[0].text(i + width/2, incorr + 2, str(incorr), ha='center', va='bottom')

# -------------------- 2. Venn diagram (overlap of correct predictions) --------------------
# Venn diagram requires matplotlib_venn
try:
    ax[1].set_title('Overlap of Correct Predictions')
    venn2(subsets=(base_only, ft_only, both), set_labels=('Base Model', 'Fine‑tuned Model'), ax=ax[1])
except NameError:
    ax[1].text(0.5, 0.5, 'Venn diagram not available\n(install matplotlib-venn)', 
               ha='center', va='center', transform=ax[1].transAxes)
    ax[1].set_title('Overlap (install matplotlib-venn)')

plt.tight_layout()
plt.savefig("correctness_venn_bar.png", dpi=150)
plt.show()

# -------------------- 3. Stacked bar of four categories --------------------
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Both', 'Only Base', 'Only Fine‑tuned', 'Neither']
counts = [both, base_only, ft_only, neither]
colors = ['#27ae60', '#3498db', '#e67e22', '#7f8c8d']

bars = ax.bar(categories, counts, color=colors)
ax.set_ylabel('Number of cases')
ax.set_title('Distribution of Correctness Across Models')

# Add value labels
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(count),
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig("correctness_distribution.png", dpi=150)
plt.show()

print("\nPlots saved as 'correctness_venn_bar.png' and 'correctness_distribution.png'.")
