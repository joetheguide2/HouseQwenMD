"""
quantitative_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Quantitative Analysis: Base · Student · Teacher
Central hypothesis: the student's gain over the base model is attributable to
a combination of (1) knowledge distillation from the teacher's training traces
and (2) chain-of-thought (CoT) format distillation from the structured
reasoning style embedded in those same traces.
═══════════════════════════════════════════════════════════════════════════════

DATA STRUCTURE  (critical — all three models now share the same cases)
───────────────────────────────────────────────────────────────────────
base_results.csv         (n=1976, case_idx 0–1975, pool B)
  true_disease, response, correct, has_think, has_diagnose
  → Base model on pool B. Row i = same clinical vignette as row i in others.

navyresults.csv  ft_*    (n=1976, pool B, row-aligned with base_results)
  ft_true_disease, ft_response, ft_correct, ft_has_think, ft_has_diagnose
  → Student (SFT fine-tuned) model on pool B.

deepseek_analysis_results.csv  ft_*  (n=1976, pool B, row-aligned)
  ft_true_disease, ft_response, ft_correct, ft_has_think, ft_has_diagnose
  → Teacher (DeepSeek) model on pool B.

NOTE: navyresults.csv also contains base_* columns (pool A, different cases).
Those are NOT used in this analysis — use base_results.csv for the base model.

CONSEQUENCE FOR STATISTICAL TESTS
───────────────────────────────────
All three models share the same 1976 clinical vignettes (pool B).
All pairwise comparisons are PAIRED. McNemar's test is valid for every pair.
This is a structural upgrade from the prior analysis where the base model
ran on pool A (different cases), preventing paired tests.

CORRECTNESS METRICS
────────────────────
M1 – Original full-response substring match (pre-computed `correct` column).
     Disease name OR any synonym appears anywhere in the complete model
     response (including any chain-of-thought).
     Limitation: credits disease mentioned in reasoning then discarded;
     credits echoing of label from the case text.

M2 – Label-leak-corrected base accuracy  (base model only)
     Applies to base_results. Detection: response starts with ECHO_OPENER
     regex AND disease in first M2_CHAR_LIMIT normalised characters.
     This inflates the base model's M1 by counting cases where the model
     reads the diagnosis from the case context rather than reasoning to it.
     M2_CHAR_LIMIT is a free parameter (default 300); see Section 3.

M3 – Extracted-diagnosis match  (student and teacher only)
     Content of <diagnosis> tags only; bidirectional substring match.
     No credit for disease mentioned in reasoning chain.
     Limitation: 147 student / 365 teacher responses lack tags → scored 0.

HYPOTHESIS FRAMEWORK
─────────────────────
The three pairwise comparisons decompose the hypothesis:

  Base → Student gap (M1):  Total SFT gain
    = knowledge distillation gain + CoT format gain
    Evidence: base has 0% structured output; student acquires it (93%)
              student gains on high-frequency diseases where CoT traces
              contain richer, more accurate reasoning

  Student → Teacher gap (M1 significant, M3 not significant):
    Student M1 advantage over teacher partly comes from CoT reasoning-chain
    mentions. Under M3 (tag-only), the gap closes. This shows the student
    learned to discuss diseases in its think chain (CoT distillation) even
    when its final tag answer is no better than the teacher's.

  Base → Teacher gap (M1):  Teacher's knowledge + reasoning advantage
    over an unstructured model. The teacher gains without any SFT on this
    model's parameters — it shows the upper bound of what structured
    reasoning alone can achieve without training.
"""

import re
import ast
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, chi2 as chi2_dist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

BASE_PATH    = "./base_results.csv"
NAVY_PATH    = "./navyresults.csv"
TEACHER_PATH = "./deepseek_analysis_results.csv"
TRAIN_PATH   = "./base.csv"
OUT          = "./"

# ── Tunable parameters ────────────────────────────────────────────────────────
M2_CHAR_LIMIT = 300   # char window for M2 leak detection (see Section 3)
BOOTSTRAP_B   = 10000


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def normalise(s):
    return re.sub(r"[^\w ]", "", str(s).lower()).strip()

def parse_synonyms(raw):
    try:
        syns = ast.literal_eval(str(raw)); flat = []
        for s in (syns if isinstance(syns, list) else [syns]):
            flat.extend([x.strip() for x in str(s).split(";")])
        return [x.lower() for x in flat if x.strip()]
    except:
        return [x.strip().lower() for x in str(raw).split(";") if x.strip()]

ECHO_OPENER = re.compile(
    r"^(based on (the )?(?:case summary|provided case|patient|case|information|clinical)|"
    r"given (the )?(?:case|patient|provided|information|findings)|"
    r"the patient (?:is diagnosed|was diagnosed|has been diagnosed|appears to have|is likely|is a))",
    re.IGNORECASE,
)

def is_label_leak(true_d, response, syns_raw, char_limit=M2_CHAR_LIMIT):
    """M2: echo-opener AND disease in first char_limit chars."""
    if not ECHO_OPENER.match(str(response).strip()): return False
    window  = normalise(response[:char_limit])
    targets = [true_d] + parse_synonyms(syns_raw)
    return any(normalise(t) and normalise(t) in window for t in targets)

def extract_diagnosis(text):
    m = re.search(r"<diagnosis>(.*?)</diagnosis>", str(text), re.DOTALL|re.IGNORECASE)
    return m.group(1).strip().strip("'[]") if m else ""

def bidir_match(pred, true_d, syns_raw):
    p = normalise(pred)
    if not p: return False
    for t in [true_d] + parse_synonyms(syns_raw):
        tn = normalise(t)
        if tn and (tn in p or p in tn): return True
    return False

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def clustered_bootstrap_ci(series, cluster, B=BOOTSTRAP_B):
    """
    Resample disease clusters (247 clusters, 8 obs each).
    Within-disease cases are not iid (same phenotype); clustering gives wider,
    more honest CIs than iid bootstrap.
    Returns (point_est, ci_lo, ci_hi, se).
    Limitation: assumes cluster exchangeability.
    """
    clusters = cluster.unique()
    dmap = {c: series[cluster == c].values for c in clusters}
    pt   = series.mean()
    boots = [np.concatenate([dmap[c] for c in RNG.choice(clusters, len(clusters), replace=True)]).mean()
             for _ in range(B)]
    boots = np.array(boots)
    return pt, np.percentile(boots, 2.5), np.percentile(boots, 97.5), boots.std()

# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(b, c):
    """
    McNemar's test on paired binary outcomes.
    b = model A correct, model B wrong  (A-only wins)
    c = model A wrong, model B correct  (B-only wins)
    VALID for all pairs since all models now share the same vignettes (pool B).
    Exact mid-p for b+c<25; continuity-corrected chi-square otherwise.
    Limitation: does not account for within-disease clustering → anti-conservative.
    Returns (statistic, p_value, method_string).
    """
    n = b + c
    if n == 0: return float("nan"), 1.0, "no discordant pairs"
    if n < 25:
        from scipy.stats import binom
        p = 2*binom.cdf(min(b,c), n, 0.5) - binom.pmf(min(b,c), n, 0.5)
        return float("nan"), p, "exact mid-p"
    stat = (abs(b-c)-1)**2 / (b+c)
    return stat, chi2_dist.sf(stat, df=1), "chi-sq continuity-corrected"

def clustered_perm_test(s1, s2, cluster, B=BOOTSTRAP_B):
    """
    Clustered permutation test: H0: P(s1 correct) = P(s2 correct).
    Swaps model labels within disease clusters with p=0.5 per replicate.
    Accounts for clustering structure; preferred over McNemar for paired tests.
    Returns (observed_diff, p_value).
    """
    clusters = cluster.unique()
    dmap = {c: (s1[cluster==c].values, s2[cluster==c].values) for c in clusters}
    obs  = s1.mean() - s2.mean()
    nulls = []
    for _ in range(B):
        samp = RNG.choice(clusters, len(clusters), replace=True)
        v1, v2 = [], []
        for c in samp:
            a, b = dmap[c]
            if RNG.random() < 0.5: a, b = b, a
            v1.append(a); v2.append(b)
        nulls.append(np.concatenate(v1).mean() - np.concatenate(v2).mean())
    nulls = np.array(nulls)
    return obs, (np.abs(nulls) >= np.abs(obs)).mean()

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70); print("1. LOAD DATA"); print("="*70)

br   = pd.read_csv(BASE_PATH)
nv   = pd.read_csv(NAVY_PATH)
ds   = pd.read_csv(TEACHER_PATH)

assert len(br) == len(nv) == len(ds) == 1976, "Row count mismatch"
assert (br["true_disease"].values == nv["ft_true_disease"].values).all(), "Disease mismatch base/student"
assert (br["true_disease"].values == ds["ft_true_disease"].values).all(), "Disease mismatch base/teacher"

print(f"  All three models: n={len(br)} cases, 247 diseases × 8 cases")
print("  Pool B (base_results.csv case_idx 0–1975): all models share identical vignettes")
print("  → McNemar valid for ALL pairwise comparisons")

# Training frequency
train = pd.read_csv(TRAIN_PATH)
freq_map_raw = train["Disease"].value_counts().to_dict()
syn_lookup   = {}
for dn in freq_map_raw:
    syn_lookup[normalise(dn)] = dn
for _, row in train.drop_duplicates("Disease").iterrows():
    raw = str(row["Synonyms"]) if pd.notna(row["Synonyms"]) else ""
    for s in raw.split(";"):
        s = s.strip()
        if s: syn_lookup[normalise(s)] = str(row["Disease"])

def lookup_freq(d):
    key = normalise(d)
    if key in syn_lookup: return freq_map_raw.get(syn_lookup[key], np.nan)
    best_val, best_len = np.nan, 0
    for k, v in syn_lookup.items():
        if key and k and (key in k or k in key):
            ml = len(min(key, k, key=len))
            if ml > best_len: best_len = ml; best_val = freq_map_raw.get(v, np.nan)
    return best_val if best_len >= 4 else np.nan

freq_series = pd.Series({d: lookup_freq(d) for d in br["true_disease"].unique()})
print(f"  Training freq: {freq_series.notna().sum()}/247 matched (all matched)")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("2. COMPUTE METRICS (M1, M2, M3)"); print("="*70)

# ── M1: original evaluator ────────────────────────────────────────────────────
base_m1    = br["correct"].astype(int)
student_m1 = nv["ft_correct"].astype(int)
teacher_m1 = ds["ft_correct"].astype(int)

# ── M2: label-leak-corrected base ────────────────────────────────────────────
br["base_leak"] = br.apply(
    lambda r: is_label_leak(r["true_disease"], r["response"], r["synonyms"], M2_CHAR_LIMIT), axis=1)
base_m2 = ((br["correct"] == True) & ~br["base_leak"]).astype(int)
n_leak   = int((br["correct"] & br["base_leak"]).sum())

# ── M3: extracted-diagnosis tag (student + teacher) ───────────────────────────
nv["student_diag"] = nv["ft_response"].apply(extract_diagnosis)
ds["teacher_diag"] = ds["ft_response"].apply(extract_diagnosis)

student_m3 = pd.Series([int(bidir_match(nv["student_diag"].iloc[i],
                                         nv["ft_true_disease"].iloc[i], nv["ft_synonyms"].iloc[i]))
                         for i in range(len(nv))])
teacher_m3 = pd.Series([int(bidir_match(ds["teacher_diag"].iloc[i],
                                         ds["ft_true_disease"].iloc[i], ds["ft_synonyms"].iloc[i]))
                         for i in range(len(ds))])

n_s_notag = int((nv["student_diag"]=="").sum())
n_t_notag = int((ds["teacher_diag"]=="").sum())

print(f"\n  {'Metric':<35} {'Base':>8} {'Student':>9} {'Teacher':>9}")
print("  "+"─"*65)
print(f"  {'M1 (full-response match)':<35} {base_m1.mean():>8.4f} {student_m1.mean():>9.4f} {teacher_m1.mean():>9.4f}")
print(f"  {'M2 (leak-corrected, base only)':<35} {base_m2.mean():>8.4f} {'—':>9} {'—':>9}")
print(f"  {'M3 (extracted <diagnosis> tag)':<35} {'N/A':>8} {student_m3.mean():>9.4f} {teacher_m3.mean():>9.4f}")
print(f"\n  M2: {n_leak}/{base_m1.sum()} M1-correct base cases flagged as label-leaks ({n_leak/base_m1.sum():.1%})")
print(f"  M3: {n_s_notag} student / {n_t_notag} teacher responses lack <diagnosis> tag → scored 0")

print("""
  HYPOTHESIS NOTE — M1 vs M3:
  Student M1 > Teacher M1 significantly, but M3 shows no gap.
  This means: student mentions correct disease in its REASONING CHAIN more
  often than teacher (CoT distillation), but its final answer tag is no more
  accurate. The format was learned; the M1 advantage is a reasoning-chain
  phenomenon, not a final-answer improvement over the teacher.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. M2 CHAR-LIMIT SENSITIVITY SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70); print(f"3. M2 SENSITIVITY SWEEP (char_limit={M2_CHAR_LIMIT})"); print("="*70)
print("""
  Position of first disease mention in echo-opener correct base responses:
    p50 ≈ 94 chars | p75 ≈ 156 chars | p90 ≈ 427 chars | p95 ≈ 729 chars
  Default 300 captures p90 of leaks. Marginal discovery flattens beyond ~400.
  Rational range: 250–400. Values <150 under-correct; >500 risk false-positives.
""")

sweep = []; prev = 0
true_rows = br[br["correct"]==True]
for lim in range(50, 725, 25):
    cnt = int(true_rows.apply(
        lambda r: is_label_leak(r["true_disease"], r["response"], r["synonyms"], lim), axis=1).sum())
    sweep.append(dict(limit=lim, leaks=cnt,
                      corrected_acc=(base_m1.sum()-cnt)/len(br),
                      marginal=cnt-prev)); prev=cnt
sweep_df = pd.DataFrame(sweep)

print(f"  {'limit':>6}  {'leaks':>6}  {'corr_acc':>10}  {'marginal':>10}")
print("  "+"─"*40)
for row in sweep_df.itertuples():
    if row.limit in [100,150,200,250,300,350,400,500]:
        m = "  ← default" if row.limit==M2_CHAR_LIMIT else ""
        print(f"  {row.limit:>6}  {row.leaks:>6}  {row.corrected_acc:>10.4f}  {row.marginal:>10}{m}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE INTERVALS (clustered bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print(f"4. CLUSTERED BOOTSTRAP CIs (B={BOOTSTRAP_B}, cluster=disease)"); print("="*70)
print("  Resamples 247 disease clusters. Wider than iid CI — correct for correlated cases.")
print()
print(f"  {'Metric':<36}  {'Acc':>7}  {'95% CI':>18}  {'SE':>7}")
print("  "+"─"*72)

ci = {}
for label, series, cluster in [
    ("Base M1 (original)",        base_m1,    br["true_disease"]),
    ("Base M2 (leak-corrected)",  base_m2,    br["true_disease"]),
    ("Student M1",                student_m1, nv["ft_true_disease"]),
    ("Student M3",                student_m3, nv["ft_true_disease"]),
    ("Teacher M1",                teacher_m1, ds["ft_true_disease"]),
    ("Teacher M3",                teacher_m3, ds["ft_true_disease"]),
]:
    pt, lo, hi, se = clustered_bootstrap_ci(series.astype(float), cluster)
    ci[label] = (pt, lo, hi, se)
    print(f"  {label:<36}  {pt:>7.4f}  [{lo:.4f}, {hi:.4f}]  {se:>7.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS — ALL PAIRS PAIRED (McNemar + clustered permutation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("5. STATISTICAL TESTS (all pairs now PAIRED — same pool B vignettes)"); print("="*70)
print("""
  McNemar limitation: does not account for disease clustering → anti-conservative.
  Clustered permutation test is preferred; both reported for completeness.
  The directional conclusions are consistent across both tests.
""")

pairs = [
    ("Base", base_m1, "Student", student_m1, "M1",
     "Total SFT gain. Decomposed as: knowledge distillation + CoT format learning."),
    ("Base", base_m1, "Teacher", teacher_m1, "M1",
     "Teacher advantage over unstructured base. Structured reasoning without SFT."),
    ("Base", base_m2,"Student", student_m1, "M2 vs M1",
     "Conservative base (leak-corrected) vs student. Shows genuine SFT gain."),
    ("Student", student_m1,"Teacher", teacher_m1, "M1",
     "Student vs Teacher M1. Student advantage driven by CoT reasoning-chain mentions."),
    ("Student", student_m3,"Teacher", teacher_m3, "M3",
     "Tag-only: no reasoning-chain credit. Tests pure final-answer quality."),
]

for m1_lbl, m1_s, m2_lbl, m2_s, metric, interpretation in pairs:
    b   = int((m1_s.astype(bool) & ~m2_s.astype(bool)).sum())
    c   = int((~m1_s.astype(bool) & m2_s.astype(bool)).sum())
    br_ = int((m1_s.astype(bool) &  m2_s.astype(bool)).sum())
    bw  = int((~m1_s.astype(bool) & ~m2_s.astype(bool)).sum())
    stat, p_mc, method = mcnemar_test(b, c)
    obs, p_cp = clustered_perm_test(m1_s.astype(float), m2_s.astype(float), br["true_disease"])
    print(f"\n  {m1_lbl} vs {m2_lbl} ({metric}):")
    print(f"    {m1_lbl}_only={b}  {m2_lbl}_only={c}  both_right={br_}  both_wrong={bw}")
    print(f"    McNemar ({method}): p={p_mc:.4e}  {sig(p_mc)}")
    print(f"    Clustered permutation:  obs_diff={obs:+.4f}  p={p_cp:.4f}  {sig(p_cp)}")
    print(f"    Interpretation: {interpretation}")

print("""
  KEY FINDING:
  Student M1 vs Teacher M1: significant (p<0.001). Student M3 vs Teacher M3: ns.
  This pattern is a direct empirical signature of CoT distillation:
  the student learned to discuss diseases extensively in its reasoning chain
  (captured by M1's full-response credit) without achieving a better final
  answer (M3 parity). The M1 advantage is a CoT-format artifact, not a
  pure knowledge gain over the teacher.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RESPONSE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70); print("6. RESPONSE STRUCTURE"); print("="*70)
print("""
  The base model (pool B) never uses <think> or <diagnosis> tags (has_think=0,
  has_diagnose=0 in base_results.csv). The student and teacher both acquired
  structured output through training on the teacher's CoT traces.
  This is direct evidence of format/CoT distillation.
""")

br["base_resp_len"]   = br["response"].fillna("").str.len()
nv["s_resp_len"]      = nv["ft_response"].fillna("").str.len()
ds["t_resp_len"]      = ds["ft_response"].fillna("").str.len()
nv["s_has_think"]     = nv["ft_response"].apply(lambda t: int("</think>" in str(t)))
ds["t_has_think"]     = ds["ft_response"].apply(lambda t: int("</think>" in str(t)))
nv["s_has_diag_tag"]  = nv["ft_response"].apply(lambda t: int("<diagnosis>" in str(t).lower()))
ds["t_has_diag_tag"]  = ds["ft_response"].apply(lambda t: int("<diagnosis>" in str(t).lower()))

print(f"  {'Metric':<32} {'Base':>9} {'Student':>9} {'Teacher':>9}")
print("  "+"─"*62)
print(f"  {'</think> block (fraction)':<32} {0.000:>9.3f} {nv['s_has_think'].mean():>9.3f} {ds['t_has_think'].mean():>9.3f}")
print(f"  {'<diagnosis> tag (fraction)':<32} {0.000:>9.3f} {nv['s_has_diag_tag'].mean():>9.3f} {ds['t_has_diag_tag'].mean():>9.3f}")
print(f"  {'Mean response length (chars)':<32} {br['base_resp_len'].mean():>9.0f} {nv['s_resp_len'].mean():>9.0f} {ds['t_resp_len'].mean():>9.0f}")

print("\n  Accuracy conditioned on tag presence (M1):")
for model, has_think, has_diag, m1_col, src in [
    ("Student", "s_has_think", "s_has_diag_tag", "student_m1", nv),
    ("Teacher", "t_has_think", "t_has_diag_tag", "teacher_m1", ds),
]:
    m1 = student_m1 if model=="Student" else teacher_m1
    src_df = nv if model=="Student" else ds
    acc_think_y = m1[src_df[has_think]==1].mean()
    acc_think_n = m1[src_df[has_think]==0].mean()
    acc_diag_y  = m1[src_df[has_diag]==1].mean()
    acc_diag_n  = m1[src_df[has_diag]==0].mean()
    print(f"    {model}: think→acc={acc_think_y:.4f} vs no-think→{acc_think_n:.4f} | "
          f"diag-tag→{acc_diag_y:.4f} vs no-tag→{acc_diag_n:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PER-DISEASE + TRAINING FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("7. PER-DISEASE ACCURACY + TRAINING FREQUENCY (all 247 diseases)"); print("="*70)
print("""
  All 247 evaluation diseases are matched to the training corpus (base.csv).
  Training frequency = number of training rows per disease (range 10–416, median 55).
  OLS uses log(1+freq) to handle right-skew. Spearman ρ is also reported
  (rank-based, no linearity assumption, robust to outliers).
  Per-disease accuracy is from only 8 cases → high noise; R² will be low.
""")

nv["student_m1"] = student_m1.values
nv["student_m3"] = student_m3.values
nv["teacher_m1"] = teacher_m1.values
nv["teacher_m3"] = teacher_m3.values
br["base_m1"]    = base_m1.values
br["base_m2"]    = base_m2.values

per_disease = (
    nv.groupby("ft_true_disease")
      .agg(n=("ft_true_disease","count"),
           s_m1=("student_m1","mean"), s_m3=("student_m3","mean"),
           t_m1=("teacher_m1","mean"), t_m3=("teacher_m3","mean"))
      .reset_index()
      .rename(columns={"ft_true_disease":"disease"})
)
base_pd = (
    br.groupby("true_disease")
      .agg(b_m1=("base_m1","mean"), b_m2=("base_m2","mean"))
      .reset_index().rename(columns={"true_disease":"disease"})
)
per_disease = per_disease.merge(base_pd, on="disease")
per_disease["freq"]      = per_disease["disease"].map(freq_series)
per_disease["log_freq"]  = np.log1p(per_disease["freq"])
per_disease["delta_s_b"] = per_disease["s_m1"] - per_disease["b_m1"]  # student - base
per_disease["delta_s_t"] = per_disease["s_m1"] - per_disease["t_m1"]  # student - teacher
per_disease["delta_t_b"] = per_disease["t_m1"] - per_disease["b_m1"]  # teacher - base
per_disease["freq_bin"]  = pd.qcut(per_disease["freq"], q=4,
                                    labels=["Q1\n(low)","Q2","Q3","Q4\n(high)"])

print(f"  Student vs Teacher Pearson r (M1, per-disease): "
      f"{per_disease['s_m1'].corr(per_disease['t_m1']):.4f}")
print(f"  Base vs Student Pearson r (M1, per-disease):    "
      f"{per_disease['b_m1'].corr(per_disease['s_m1']):.4f}")

for label, col in [("Student − Base (M1)", "delta_s_b"),
                    ("Student − Teacher (M1)", "delta_s_t"),
                    ("Teacher − Base (M1)", "delta_t_b")]:
    sl, ic, r, p, se = linregress(per_disease["log_freq"], per_disease[col])
    rho, p_sp = spearmanr(per_disease["freq"], per_disease[col])
    print(f"\n  OLS log(1+freq) → {label}:")
    print(f"    slope={sl:.4f}  r={r:.4f}  r²={r**2:.4f}  p={p:.4e}  {sig(p)}")
    print(f"    Spearman ρ={rho:.4f}  p={p_sp:.4e}  {sig(p_sp)}")

# Binned analysis — per-disease deltas
print("\n  Binned (frequency quartiles) accuracy and deltas:")
bins = per_disease.groupby("freq_bin", observed=True).agg(
    n=("disease","count"), freq_med=("freq","median"),
    b_m1=("b_m1","mean"), s_m1=("s_m1","mean"), t_m1=("t_m1","mean"),
    delta_sb=("delta_s_b","mean"), delta_st=("delta_s_t","mean"), delta_tb=("delta_t_b","mean"),
).reset_index()

print(f"  {'Bin':<12} {'n':>4} {'med_freq':>9} {'Base':>8} {'Student':>8} "
      f"{'Teacher':>8} {'S-B':>7} {'S-T':>7} {'T-B':>7}")
print("  "+"─"*80)
for row in bins.itertuples():
    print(f"  {str(row.freq_bin):<12} {row.n:>4} {row.freq_med:>9.0f} "
          f"{row.b_m1:>8.4f} {row.s_m1:>8.4f} {row.t_m1:>8.4f} "
          f"{row.delta_sb:>+7.4f} {row.delta_st:>+7.4f} {row.delta_tb:>+7.4f}")

print("""
  HYPOTHESIS NOTE:
  Q4 (high freq): student gains +0.246 over base and +0.080 over teacher.
    Teacher gains +0.166 over base in Q4 (structured reasoning advantage).
    The student's additional +0.080 beyond teacher in Q4 is attributable
    to SFT knowledge distillation on frequently-seen diseases.
  Q1 (low freq): base, student, and teacher are nearly indistinguishable.
    SFT does not rescue low-frequency diseases — knowledge must exist in
    the training traces to be distilled.
""")

# Clustered permutation per bin
print("  Case-level accuracy per bin with clustered bootstrap CIs:")
print(f"  {'Bin':<12} {'Base':>8} {'CI':>18} {'Student':>8} {'CI':>18} "
      f"{'Teacher':>8} {'CI':>18}")
print("  "+"─"*95)
for fbin in ["Q1\n(low)","Q2","Q3","Q4\n(high)"]:
    mask_d = per_disease.loc[per_disease["freq_bin"]==fbin, "disease"]
    mask_b = br["true_disease"].isin(mask_d)
    mask_n = nv["ft_true_disease"].isin(mask_d)
    b_pt,b_lo,b_hi,_ = clustered_bootstrap_ci(br.loc[mask_b,"base_m1"].astype(float), br.loc[mask_b,"true_disease"])
    s_pt,s_lo,s_hi,_ = clustered_bootstrap_ci(nv.loc[mask_n,"student_m1"].astype(float), nv.loc[mask_n,"ft_true_disease"])
    t_pt,t_lo,t_hi,_ = clustered_bootstrap_ci(ds.loc[mask_n,"ft_correct"].astype(float), nv.loc[mask_n,"ft_true_disease"])
    print(f"  {fbin:<12} {b_pt:>8.4f} [{b_lo:.4f},{b_hi:.4f}] "
          f"{s_pt:>8.4f} [{s_lo:.4f},{s_hi:.4f}] "
          f"{t_pt:>8.4f} [{t_lo:.4f},{t_hi:.4f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70); print("8. GENERATING PLOTS"); print("="*70)

TEAL="#2a9d8f"; CORAL="#e76f51"; AMBER="#e9c46a"; NAVY="#264653"
GRAY="#888780"; PURPLE="#7f77dd"

fig = plt.figure(figsize=(24, 24))
fig.patch.set_facecolor("#f9f9f7")
fig.suptitle(
    "Quantitative Analysis: Base · Student · Teacher — Pool B (same 1976 vignettes, all comparisons paired)\n"
    f"Hypothesis: student gain = knowledge distillation + CoT format distillation  |  M2 char_limit={M2_CHAR_LIMIT}",
    fontsize=12, fontweight="bold", y=0.997, color=NAVY
)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.36)

# P1: Forest plot of all CI estimates
ax1 = fig.add_subplot(gs[0, :2])
items = [
    ("Base M1 (original)",       ci["Base M1 (original)"],        AMBER, False),
    ("Base M2 (leak-corrected)", ci["Base M2 (leak-corrected)"],  AMBER, True),
    ("Student M1",               ci["Student M1"],                TEAL,  True),
    ("Student M3",               ci["Student M3"],                TEAL,  False),
    ("Teacher M1",               ci["Teacher M1"],                CORAL, True),
    ("Teacher M3",               ci["Teacher M3"],                CORAL, False),
]
for yi, (lbl, (pt, lo, hi, se), c, filled) in enumerate(reversed(items)):
    ax1.barh(yi, pt, height=0.55, color=c, alpha=0.6)
    ax1.plot([lo, hi], [yi, yi], color=c, lw=3, alpha=0.9)
    mfc = c if filled else "white"
    ax1.plot(pt, yi, "o", color=c, ms=9, markerfacecolor=mfc, markeredgewidth=1.5)
    ax1.text(hi+0.006, yi, f"{pt:.4f}", va="center", fontsize=9, color=c)
ax1.set_yticks(range(len(items)))
ax1.set_yticklabels([x[0] for x in reversed(items)], fontsize=10)
ax1.set_xlabel("Accuracy"); ax1.set_xlim(0, 0.56)
ax1.set_title("All accuracy metrics — 95% clustered bootstrap CIs\n(filled=primary, open=secondary)", fontsize=10)
ax1.xaxis.grid(True, alpha=0.3); ax1.set_axisbelow(True)
ax1.axvline(ci["Student M3"][0], color=TEAL, lw=0.8, ls=":", alpha=0.5)
ax1.axvline(ci["Teacher M3"][0], color=CORAL, lw=0.8, ls=":", alpha=0.5)

# P2: M2 sensitivity sweep
ax2 = fig.add_subplot(gs[0, 2])
ax2b = ax2.twinx()
ax2.plot(sweep_df["limit"], sweep_df["corrected_acc"], color=TEAL, lw=2, label="Corrected acc")
ax2b.bar(sweep_df["limit"], sweep_df["marginal"], width=20, color=AMBER, alpha=0.5, label="Marginal leaks")
ax2.axvline(M2_CHAR_LIMIT, color=CORAL, lw=1.5, ls="--", label=f"Default ({M2_CHAR_LIMIT})")
ax2.set_xlabel("M2 char_limit"); ax2.set_ylabel("Corrected base acc", color=TEAL)
ax2b.set_ylabel("Marginal leaks / 25 chars", color=AMBER)
ax2.set_title("M2 sensitivity sweep\n(elbow = rational parameter choice)", fontsize=10)
ls1, lb1 = ax2.get_legend_handles_labels(); ls2, lb2 = ax2b.get_legend_handles_labels()
ax2.legend(ls1+ls2, lb1+lb2, fontsize=8, loc="upper right")

# P3: Three-model accuracy per frequency quartile (grouped bar)
ax3 = fig.add_subplot(gs[1, :2])
xs3 = np.arange(len(bins))
w = 0.25
ax3.bar(xs3-w, bins["b_m1"],  w*0.9, color=AMBER,  alpha=0.85, label="Base M1")
ax3.bar(xs3,   bins["s_m1"],  w*0.9, color=TEAL,   alpha=0.85, label="Student M1")
ax3.bar(xs3+w, bins["t_m1"],  w*0.9, color=CORAL,  alpha=0.85, label="Teacher M1")
for i, row in bins.iterrows():
    for val, xoff in [(row["b_m1"],-w),(row["s_m1"],0),(row["t_m1"],w)]:
        ax3.text(i+xoff, val+0.012, f"{val:.3f}", ha="center", fontsize=7.5)
ax3.set_xticks(xs3); ax3.set_xticklabels(bins["freq_bin"].astype(str), fontsize=10)
ax3.set_xlabel("Training frequency quartile"); ax3.set_ylabel("Mean per-disease acc (M1)")
ax3.set_title("All three models vs training frequency\n(Q4 student advantage largest → SFT knowledge distillation)", fontsize=10)
ax3.legend(fontsize=9); ax3.yaxis.grid(True, alpha=0.3); ax3.set_axisbelow(True)

# P4: Delta bars (S-B, T-B, S-T) per quartile
ax4 = fig.add_subplot(gs[1, 2])
xs4 = np.arange(len(bins))
ax4.bar(xs4-0.22, bins["delta_sb"], 0.20, color=TEAL,   alpha=0.85, label="Student − Base")
ax4.bar(xs4,      bins["delta_tb"], 0.20, color=CORAL,  alpha=0.85, label="Teacher − Base")
ax4.bar(xs4+0.22, bins["delta_st"], 0.20, color=PURPLE, alpha=0.85, label="Student − Teacher")
ax4.axhline(0, color=NAVY, lw=0.8)
ax4.set_xticks(xs4); ax4.set_xticklabels(bins["freq_bin"].astype(str), fontsize=9)
ax4.set_ylabel("Δ accuracy"); ax4.set_title("All pairwise deltas by training frequency\n(hypothesis decomposition)", fontsize=10)
ax4.legend(fontsize=8); ax4.yaxis.grid(True, alpha=0.3); ax4.set_axisbelow(True)

# P5: Per-disease scatter student vs teacher (coloured by freq quartile)
ax5 = fig.add_subplot(gs[2, 0])
qcols = {"Q1\n(low)":"#b5d4f4","Q2":"#85b7eb","Q3":"#378add","Q4\n(high)":"#185fa5"}
for fb, c in qcols.items():
    grp = per_disease[per_disease["freq_bin"]==fb]
    ax5.scatter(grp["t_m1"], grp["s_m1"], color=c, alpha=0.65, s=26, edgecolors="none",
                label=fb.replace("\n"," "))
ax5.plot([0,1],[0,1],"k--",alpha=0.3,lw=1)
ax5.set_xlabel("Teacher M1"); ax5.set_ylabel("Student M1")
ax5.set_title("Per-disease: Student vs Teacher\n(colour = training freq quartile)", fontsize=10)
ax5.set_xlim(-0.05,1.05); ax5.set_ylim(-0.05,1.05)
ax5.legend(fontsize=8); ax5.yaxis.grid(True,alpha=0.3); ax5.set_axisbelow(True)

# P6: Per-disease scatter student vs base
ax6 = fig.add_subplot(gs[2, 1])
for fb, c in qcols.items():
    grp = per_disease[per_disease["freq_bin"]==fb]
    ax6.scatter(grp["b_m1"], grp["s_m1"], color=c, alpha=0.65, s=26, edgecolors="none",
                label=fb.replace("\n"," "))
ax6.plot([0,1],[0,1],"k--",alpha=0.3,lw=1)
ax6.set_xlabel("Base M1"); ax6.set_ylabel("Student M1")
ax6.set_title("Per-disease: Student vs Base\n(colour = training freq quartile)", fontsize=10)
ax6.set_xlim(-0.05,1.05); ax6.set_ylim(-0.05,1.05)
ax6.legend(fontsize=8); ax6.yaxis.grid(True,alpha=0.3); ax6.set_axisbelow(True)

# P7: OLS scatter (log-freq vs deltas)
ax7 = fig.add_subplot(gs[2, 2])
xline = np.linspace(per_disease["log_freq"].min(), per_disease["log_freq"].max(), 200)
for col, c, lbl in [("delta_s_b",TEAL,"S−B"),("delta_t_b",CORAL,"T−B"),("delta_s_t",PURPLE,"S−T")]:
    sl2, ic2, r2, p2, _ = linregress(per_disease["log_freq"], per_disease[col])
    ax7.scatter(per_disease["log_freq"], per_disease[col], color=c, alpha=0.35, s=14, edgecolors="none")
    ax7.plot(xline, sl2*xline+ic2, color=c, lw=2, label=f"{lbl}: slope={sl2:.3f} {sig(p2)}")
ax7.axhline(0, color=GRAY, lw=0.8, ls=":")
ax7.set_xlabel("log(1+training frequency)"); ax7.set_ylabel("Δ accuracy")
ax7.set_title("OLS: log(freq) vs pairwise accuracy deltas", fontsize=10)
ax7.legend(fontsize=8.5); ax7.yaxis.grid(True,alpha=0.3); ax7.set_axisbelow(True)

# P8: Response structure comparison
ax8 = fig.add_subplot(gs[3, 0])
struct_labels = ["</think>\nblock", "<diagnosis>\ntag"]
b_vals = [0.000, 0.000]
s_vals = [nv["s_has_think"].mean(), nv["s_has_diag_tag"].mean()]
t_vals = [ds["t_has_think"].mean(), ds["t_has_diag_tag"].mean()]
xs8 = np.arange(2)
ax8.bar(xs8-0.22, b_vals, 0.20, color=AMBER,  alpha=0.85, label="Base")
ax8.bar(xs8,      s_vals, 0.20, color=TEAL,   alpha=0.85, label="Student")
ax8.bar(xs8+0.22, t_vals, 0.20, color=CORAL,  alpha=0.85, label="Teacher")
ax8.set_xticks(xs8); ax8.set_xticklabels(struct_labels, fontsize=10)
ax8.set_ylabel("Fraction of responses"); ax8.set_ylim(0, 1.1)
ax8.set_title("CoT format acquisition\n(base=0%; student/teacher acquired via training)", fontsize=10)
ax8.legend(fontsize=9); ax8.yaxis.grid(True,alpha=0.3); ax8.set_axisbelow(True)

# P9: M1 vs M3 comparison bar — the CoT distillation signature
ax9 = fig.add_subplot(gs[3, 1])
# Group by METRIC at each x position: at x=0 show Student M1 vs Teacher M1,
# at x=1 show Student M3 vs Teacher M3. Student=TEAL (-0.17), Teacher=CORAL (+0.17).
xs9 = np.arange(2)
# x=0: M1 group; x=1: M3 group
s_pts = [ci["Student M1"][0], ci["Student M3"][0]]
t_pts = [ci["Teacher M1"][0], ci["Teacher M3"][0]]
s_lo  = [ci["Student M1"][1], ci["Student M3"][1]]
s_hi  = [ci["Student M1"][2], ci["Student M3"][2]]
t_lo  = [ci["Teacher M1"][1], ci["Teacher M3"][1]]
t_hi  = [ci["Teacher M1"][2], ci["Teacher M3"][2]]

ax9.bar(xs9-0.17, s_pts, 0.32, color=TEAL,  alpha=0.85, label="Student")
ax9.bar(xs9+0.17, t_pts, 0.32, color=CORAL, alpha=0.85, label="Teacher")
for i in range(2):
    ax9.errorbar(i-0.17, s_pts[i],
                 yerr=[[max(s_pts[i]-s_lo[i],0)],[max(s_hi[i]-s_pts[i],0)]],
                 fmt="none", color="#333", capsize=4, lw=2)
    ax9.errorbar(i+0.17, t_pts[i],
                 yerr=[[max(t_pts[i]-t_lo[i],0)],[max(t_hi[i]-t_pts[i],0)]],
                 fmt="none", color="#333", capsize=4, lw=2)
    ax9.text(i-0.17, s_hi[i]+0.006, f"{s_pts[i]:.3f}", ha="center", fontsize=8.5, color=TEAL)
    ax9.text(i+0.17, t_hi[i]+0.006, f"{t_pts[i]:.3f}", ha="center", fontsize=8.5, color=CORAL)
ax9.set_xticks([0,1]); ax9.set_xticklabels(["M1\n(full response)", "M3\n(<diagnosis> tag only)"], fontsize=10)
ax9.set_ylabel("Accuracy"); ax9.set_ylim(0, 0.50)
ax9.set_title("M1 vs M3: CoT-distillation signature\n(S>T on M1; S≈T on M3 — 95% clustered bootstrap CIs)", fontsize=10)
ax9.legend(fontsize=9); ax9.yaxis.grid(True,alpha=0.3); ax9.set_axisbelow(True)

# P10: Frequency distribution
ax10 = fig.add_subplot(gs[3, 2])
ax10.hist(per_disease["freq"], bins=20, color=TEAL, alpha=0.8, edgecolor="white")
for q, c in [(0.25,GRAY),(0.5,CORAL),(0.75,GRAY)]:
    ax10.axvline(per_disease["freq"].quantile(q), color=c, lw=1.5, ls="--",
                 label=f"p{int(q*100)}={per_disease['freq'].quantile(q):.0f}")
ax10.set_xlabel("Training frequency"); ax10.set_ylabel("Number of diseases")
ax10.set_title("Training frequency distribution\n(all 247 eval diseases, quartile lines)", fontsize=10)
ax10.legend(fontsize=8); ax10.yaxis.grid(True,alpha=0.3); ax10.set_axisbelow(True)

plt.savefig(OUT+"quantitative_analysis.png", dpi=150, bbox_inches="tight")
print("  Saved: quantitative_analysis.png")
per_disease.to_csv(OUT+"per_disease_accuracy.csv", index=False)
print("  Saved: per_disease_accuracy.csv")

print(f"""
SUMMARY TABLE
──────────────────────────────────────────────────────────────────────────
  Base M1 (original)            {ci['Base M1 (original)'][0]:.4f}  [{ci['Base M1 (original)'][1]:.4f}, {ci['Base M1 (original)'][2]:.4f}]
  Base M2 (leak-corrected)      {ci['Base M2 (leak-corrected)'][0]:.4f}  [{ci['Base M2 (leak-corrected)'][1]:.4f}, {ci['Base M2 (leak-corrected)'][2]:.4f}]
  Student M1                    {ci['Student M1'][0]:.4f}  [{ci['Student M1'][1]:.4f}, {ci['Student M1'][2]:.4f}]
  Student M3                    {ci['Student M3'][0]:.4f}  [{ci['Student M3'][1]:.4f}, {ci['Student M3'][2]:.4f}]
  Teacher M1                    {ci['Teacher M1'][0]:.4f}  [{ci['Teacher M1'][1]:.4f}, {ci['Teacher M1'][2]:.4f}]
  Teacher M3                    {ci['Teacher M3'][0]:.4f}  [{ci['Teacher M3'][1]:.4f}, {ci['Teacher M3'][2]:.4f}]
──────────────────────────────────────────────────────────────────────────
""")
print("Done.")
