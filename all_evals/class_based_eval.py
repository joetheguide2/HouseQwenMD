import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm

# =============================================================================
df_eval_cot = pd.read_csv('navyresults.csv')
df_eval_sft = pd.read_csv('commonv2.csv')
df_eval_cot_oneshot = pd.read_csv('deepseek_analysis_results.csv')
#df_eval_faux_accuracy_cases = pd.read_csv('./casestudyfauxaccuracy.csv')

train_freq = pd.read_parquet('./base.parquet')

df_eval = df_eval_cot.copy()
df_eval['base_correct'] = df_eval_sft['ft_correct'].astype(bool)
df_eval['ft_correct'] = df_eval_cot['ft_correct'].astype(bool)

#df_eval = df_eval.loc[~ df_eval_faux_accuracy_cases['ft_correct']]


# =============================================================================
# 2. Build mapping from disease name to training frequency
# =============================================================================
def parse_synonyms(syn):
    if isinstance(syn, str):
        for delim in [',', ';', '|', '/']:
            syn = syn.replace(delim, ',')
        return [s.strip() for s in syn.split(',') if s.strip()]
    elif isinstance(syn, (list, np.ndarray)):
        return [str(s).strip() for s in syn if pd.notna(s)]
    else:
        print(f"Warning: unexpected type {type(syn)}: {syn}")
        return []

freq_map = {}
for _, row in train_freq.iterrows():
    synonyms = parse_synonyms(row['Disease'])
    freq = row['Frequency']
    for name in synonyms:
        if name in freq_map:
            freq_map[name] = max(freq_map[name], freq)
        else:
            freq_map[name] = freq

# =============================================================================
# 3. Merge evaluation data with training frequency
# =============================================================================
df_eval['train_freq'] = df_eval['base_true_disease'].map(freq_map)

before = len(df_eval)
df_merged = df_eval.dropna(subset=['train_freq']).copy()

# Assume df_merged has columns: base_true_disease, train_freq, base_correct, ft_correct
# Compute per-disease improvement and log frequency
per_disease = df_merged.groupby('base_true_disease').agg(
    train_freq=('train_freq', 'first'),  # same for all samples of a disease
    base_recall=('base_correct', 'mean'),
    ft_recall=('ft_correct', 'mean'),
    eval_samples=('base_correct', 'count')
).reset_index()
per_disease['improvement'] = per_disease['ft_recall'] - per_disease['base_recall']
per_disease['log_freq'] = np.log10(per_disease['train_freq'] + 1)  # +1 to avoid log(0)
per_disease  = per_disease[per_disease['eval_samples'] > 4]


# Linear regression on log frequency
X = sm.add_constant(per_disease['log_freq'])
y = per_disease['improvement']
model = sm.OLS(y, X).fit()
print(model.summary())

# Plot with regression line and confidence interval
plt.figure(figsize=(10,6))
plt.scatter(per_disease['log_freq'], per_disease['improvement'], 
            alpha=0.3, s=per_disease['eval_samples']*5, c='blue')
# Sort for line
x_sorted = np.sort(per_disease['log_freq'])
X_pred = sm.add_constant(x_sorted)
pred = model.get_prediction(X_pred)
pred_ci = pred.conf_int(alpha=0.05)
plt.plot(x_sorted, pred.predicted_mean, 'g-', lw=2, label='OLS fit')
plt.fill_between(x_sorted, pred_ci[:,0], pred_ci[:,1], color='green', alpha=0.2, label='95% CI')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('log10(Training frequency + 1)')
plt.ylabel('Improvement')
plt.title('Linear Fit on log(Frequency) with 95% Confidence Band')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('linear_log_fit.png', dpi=150)
plt.show()

# Assuming all_results from previous threshold analysis exists
# Apply a moving average or LOWESS to the point estimates to reduce noise

all_results = pd.read_csv('./performance_with_ci.csv')

thresh = all_results['min_train_samples'].values
delta = all_results['delta_acc'].values
low = all_results['delta_acc_ci_low'].values
high = all_results['delta_acc_ci_high'].values

# Option 1: Plot the raw step-like curve with fill_between
plt.figure(figsize=(10,6))
plt.plot(thresh, delta, 'g-', lw=1, label='Δ Accuracy')
plt.fill_between(thresh, low, high, color='green', alpha=0.2, label='95% CI')
plt.xlabel('Minimum training samples per disease')
plt.ylabel('Improvement')
plt.title('Continuous Threshold Analysis with 95% CI')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('continuous_threshold.png', dpi=150)
plt.show()


bin_diseases = df_merged[(df_merged['train_freq'] >= 16) & (df_merged['train_freq'] <= 20)]
bin_per_disease = bin_diseases.groupby('base_true_disease').agg(
    train_freq=('train_freq', 'first'),
    base_recall=('base_correct', 'mean'),
    ft_recall=('ft_correct', 'mean'),
    eval_samples=('base_correct', 'count')
).reset_index()
bin_per_disease['improvement'] = bin_per_disease['ft_recall'] - bin_per_disease['base_recall']
bin_per_disease = bin_per_disease.sort_values('improvement')
print(bin_per_disease.head(10))  # worst degradations
print(bin_per_disease.tail(10))  # best improvements

# =============================================================================
# 1. Load and prepare data (assuming df_merged already exists from earlier)
# =============================================================================
# If starting from scratch, run the earlier code to create df_merged.
# df_merged must have columns: base_true_disease, train_freq, base_correct, ft_correct.

# For this example, we assume df_merged is already in memory.

# =============================================================================
# 2. Compute per-disease statistics
# =============================================================================
# Sort by training frequency to create quantile bins
per_disease_sorted = per_disease.sort_values('train_freq').reset_index(drop=True)

# =============================================================================
# 3. Create quantile bins (e.g., 10 bins with roughly equal number of diseases)
# =============================================================================
n_bins = 10
per_disease_sorted['quantile_bin'] = pd.qcut(
    per_disease_sorted['train_freq'], 
    q=n_bins, 
    labels=False, 
    duplicates='drop'  # this may reduce number of bins if duplicates exist
)

# Check how many bins we actually got
actual_bins = per_disease_sorted['quantile_bin'].nunique()
print(f"Created {actual_bins} bins with approximately equal number of diseases.")

# =============================================================================
# 4. For each bin, compute overall metrics and bootstrap CIs
# =============================================================================
n_boot = 10000
rng = np.random.default_rng(42)
results = []

# We'll also need the original sample-level data for each bin,
# so we merge bin labels back to df_merged.
bin_labels = per_disease_sorted[['base_true_disease', 'quantile_bin']]
df_merged = df_merged.merge(bin_labels, on='base_true_disease', how='left')

unique_bins = sorted(df_merged['quantile_bin'].dropna().unique())

print("\nBootstrapping quantile bins...")
for bin_id in tqdm(unique_bins):
    sub = df_merged[df_merged['quantile_bin'] == bin_id]
    diseases_in_bin = sub['base_true_disease'].unique()
    n_diseases = len(diseases_in_bin)
    n_samples = len(sub)

    # Point estimates
    base_acc = sub['base_correct'].mean()
    ft_acc = sub['ft_correct'].mean()
    delta_acc = ft_acc - base_acc

    per_disease_bin = sub.groupby('base_true_disease').agg(
        base_recall=('base_correct', 'mean'),
        ft_recall=('ft_correct', 'mean')
    )
    base_macro = per_disease_bin['base_recall'].mean()
    ft_macro = per_disease_bin['ft_recall'].mean()
    delta_macro = ft_macro - base_macro

    # Bootstrap for accuracy (sample-level)
    boot_delta_acc = []
    indices = np.arange(n_samples)
    for _ in range(n_boot):
        idx = rng.choice(indices, size=n_samples, replace=True)
        boot_base = sub['base_correct'].iloc[idx].mean()
        boot_ft = sub['ft_correct'].iloc[idx].mean()
        boot_delta_acc.append(boot_ft - boot_base)
    ci_delta_acc = np.percentile(boot_delta_acc, [2.5, 97.5])

    # Bootstrap for macro recall (disease-level)
    dis_indices = np.arange(n_diseases)
    base_recalls = per_disease_bin['base_recall'].values
    ft_recalls = per_disease_bin['ft_recall'].values
    boot_delta_macro = []
    for _ in range(n_boot):
        idx = rng.choice(dis_indices, size=n_diseases, replace=True)
        boot_base = base_recalls[idx].mean()
        boot_ft = ft_recalls[idx].mean()
        boot_delta_macro.append(boot_ft - boot_base)
    ci_delta_macro = np.percentile(boot_delta_macro, [2.5, 97.5])

    # Store bin range for labeling (min and max training freq in bin)
    bin_min = per_disease_sorted[per_disease_sorted['quantile_bin'] == bin_id]['train_freq'].min()
    bin_max = per_disease_sorted[per_disease_sorted['quantile_bin'] == bin_id]['train_freq'].max()

    results.append({
        'bin_id': bin_id,
        'bin_range': f"{bin_min}-{bin_max}",
        'n_diseases': n_diseases,
        'n_eval_samples': n_samples,
        'base_acc': base_acc,
        'ft_acc': ft_acc,
        'delta_acc': delta_acc,
        'delta_acc_low': ci_delta_acc[0],
        'delta_acc_high': ci_delta_acc[1],
        'base_macro': base_macro,
        'ft_macro': ft_macro,
        'delta_macro': delta_macro,
        'delta_macro_low': ci_delta_macro[0],
        'delta_macro_high': ci_delta_macro[1],
    })

quantile_results = pd.DataFrame(results).sort_values('bin_id')

# =============================================================================
# 5. Display and save
# =============================================================================
print("\n=== Quantile‑Based Bins (Equal Disease Count) ===")
print(quantile_results.to_string(index=False))

quantile_results.to_csv('quantile_bins_analysis.csv', index=False)

# =============================================================================
# 6. Plot
# =============================================================================
x = np.arange(len(quantile_results))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, quantile_results['delta_acc'], width,
               yerr=[quantile_results['delta_acc'] - quantile_results['delta_acc_low'],
                     quantile_results['delta_acc_high'] - quantile_results['delta_acc']],
               capsize=3, label='Δ Accuracy', color='green', alpha=0.8)

bars2 = ax.bar(x + width/2, quantile_results['delta_macro'], width,
               yerr=[quantile_results['delta_macro'] - quantile_results['delta_macro_low'],
                     quantile_results['delta_macro_high'] - quantile_results['delta_macro']],
               capsize=3, label='Δ Macro Recall', color='purple', alpha=0.8)

ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Training frequency bin (quantile)')
ax.set_ylabel('Improvement (Fine‑tuned - Base)')
ax.set_title('Improvement by Quantile‑Based Training Frequency Bin with 95% CI')
ax.set_xticks(x)
ax.set_xticklabels(quantile_results['bin_range'], rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('quantile_bins_improvement.png', dpi=150)
plt.show()

print("\nAnalysis complete. Results saved to 'quantile_bins_analysis.csv' and plot to 'quantile_bins_improvement.png'.")
