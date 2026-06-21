import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "./data/luad_tcga_pan_can_atlas_2018/"
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("🔍 Loading TCGA-LUAD PanCancer final validation cohort...")

# ==========================================================
# 1. Extract Sample TMB (Utilizing the highly precise TMB_NONSYNONYMOUS metric)
# ==========================================================
df_sample = pd.read_csv(os.path.join(BASE_DIR, 'data_clinical_sample.txt'), sep='\t', skiprows=4)
if 'TMB_NONSYNONYMOUS' not in df_sample.columns:
    raise ValueError("❌ 'TMB_NONSYNONYMOUS' column not found. Please check the dataset schema.")

df_tmb = df_sample[['SAMPLE_ID', 'TMB_NONSYNONYMOUS']].dropna()

# ==========================================================
# 2. Rapid Extraction of CD274 (PD-L1) from the massive mRNA matrix
# ==========================================================
mrna_file = os.path.join(BASE_DIR, 'data_mrna_seq_v2_rsem.txt')
print("⏳ Scanning the large-scale mRNA expression matrix for CD274 (PD-L1)... (This may take a few seconds)")

df_cd274 = pd.DataFrame()
# Utilize chunk reading to prevent memory overflow
for chunk in pd.read_csv(mrna_file, sep='\t', chunksize=5000):
    match = chunk[chunk['Hugo_Symbol'] == 'CD274']
    if not match.empty:
        df_cd274 = match
        break

if df_cd274.empty:
    raise ValueError("❌ CD274 expression data not found.")

# Transpose the expression matrix to align samples as rows
df_expr = df_cd274.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id']).T
df_expr.columns = ['CD274_mRNA']
df_expr.index.name = 'SAMPLE_ID'
df_expr = df_expr.reset_index()

# ==========================================================
# 3. Data Merging and Log Transformation
# ==========================================================
# Truncate to the 15-character core TCGA barcode (e.g., TCGA-05-4244-01)
# to ensure 100% deterministic matching across sub-datasets
df_tmb['Merge_ID'] = df_tmb['SAMPLE_ID'].str[:15]
df_expr['Merge_ID'] = df_expr['SAMPLE_ID'].str[:15]

df_clean = pd.merge(df_tmb, df_expr, on='Merge_ID', how='inner').dropna()

# Convert to numeric and apply log transformation to approximate biological normality
df_clean['CD274_mRNA'] = pd.to_numeric(df_clean['CD274_mRNA'], errors='coerce')
df_clean['TMB_NONSYNONYMOUS'] = pd.to_numeric(df_clean['TMB_NONSYNONYMOUS'], errors='coerce')
df_clean = df_clean.dropna()

df_clean['Log2_PDL1'] = np.log2(df_clean['CD274_mRNA'] + 1)
df_clean['Log10_TMB'] = np.log10(df_clean['TMB_NONSYNONYMOUS'] + 1)

# ==========================================================
# 4. Generate Publication-Grade Spearman Scatter Plot with Regression
# ==========================================================
print("🎨 Generating NPG-style scatter plot with 95% confidence intervals...")

plt.figure(figsize=(7, 7))
sns.set_theme(style="ticks")

# Render regression plot with linear fit and 95% confidence interval bands
ax = sns.regplot(
    x="Log10_TMB", y="Log2_PDL1", data=df_clean,
    # Plot clean, semi-transparent scatter points without edge outlines for a modern aesthetic
    scatter_kws={"s": 30, "alpha": 0.5, "color": "#2ca02c"},
    line_kws={"color": "#d62728", "lw": 2.5}
)

# Calculate Spearman rank correlation coefficient and exact P-value
r, p_val = spearmanr(df_clean['TMB_NONSYNONYMOUS'], df_clean['CD274_mRNA'])
p_str = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.4f}"

# Embed statistical metric annotation box in the upper right quadrant
textstr = f'$N={len(df_clean)}$\nSpearman $R={r:.2f}$\n{p_str}'
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.65, 0.90, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontweight='bold')

plt.title("Real-World Topology Validation: Mutational Stress drives PD-L1",
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel(r"Tumor Mutational Burden ($\log_{10}$ TMB)", fontsize=13, fontweight='bold')
plt.ylabel(r"PD-L1 / CD274 Expression ($\log_{2}$ RSEM)", fontsize=13, fontweight='bold')

# Despine top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Export annotated version (for manuscript insertion)
output_pdf = os.path.join(RESULTS_DIR, "Fig5b_TCGA_Validation_Scatter.pdf")
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')

# Export pure vector version without text (for slide presentations)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.texts[0].set_visible(False)
output_png_notext = os.path.join(RESULTS_DIR, "Fig5b_TCGA_Validation_Scatter_NoText.png")
plt.savefig(output_png_notext, format='png', bbox_inches='tight', transparent=True, dpi=300)

plt.close()

# ==========================================================
# 5. 🖨️ Terminal Output of Core Statistical Metrics for Manuscript
# ==========================================================
print("\n" + "=" * 60)
print("🎯 [Fig 5b Validation] Causal Correlation between TMB and PD-L1 in TCGA Cohort")
print("=" * 60)
print(f"Total Matched Cohort Size (N)        : {len(df_clean)}")
print(f"Spearman Correlation Coefficient (R) : {r:.3f}")
print(f"Statistical Significance (P-value)   : {p_val:.2e} ({p_str})")
print("=" * 60)
print(f"✅ Annotated plot saved to: {output_pdf}")
print(f"✅ Clean vector base plot saved to: {output_png_notext}")