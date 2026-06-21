"""
Table 1 Generator: Pure Causal Effects via Pearl's Do-calculus.

This script directly calculates the unbiased Causal Hazard Ratios (HR)
for Table 1. It utilizes the Minimal Sufficient Adjustment Sets (MSAS)
derived from the 11-edge Global Consensus DAG to systematically block
all back-door confounding paths for each specific exposure-outcome pair.
"""

import os
import pandas as pd
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
COHORT_PATH = "./data/synthetic_cohort_n3000_seed2026.csv"
RESULTS_DIR = "./results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

df = pd.read_csv(COHORT_PATH)

# ==========================================
# 2. Define Causal Pathways & MSAS (Based on 11-edge DAG)
# ==========================================
# 这里的 MSAS 严格遵循我们在 LaTeX 中描写的后门变量集
scenarios = [
    {
        "Group": "Dynamic Mediation",
        "Exposure": "M8_ctDNA_Clearance",
        "Outcome": "Y_PFS_TKI_All",
        "MSAS": ["X8_ctDNA_Positive", "E1_Brain_Met"],
        "Name": "ctDNA Clearance -> Targeted PFS"
    },
    {
        "Group": "Baseline Molecular Burden",
        "Exposure": "X8_ctDNA_Positive",
        "Outcome": "Y_PFS_TKI_All",
        "MSAS": ["E1_Brain_Met", "X3_Age", "X1_Gender"],
        "Name": "Baseline ctDNA(+) -> Targeted PFS"
    },
    {
        "Group": "Baseline Molecular Burden",
        "Exposure": "X2_TMB",
        "Outcome": "Y_OS_IO",
        "MSAS": ["E2_Liver_Met", "X3_Age"],
        "Name": "Tumor Mutational Burden -> Immunotherapy OS"
    },
    {
        "Group": "Macroscopic Organ Metastasis",
        "Exposure": "E1_Brain_Met",
        "Outcome": "Y_PFS_TKI_All",
        "MSAS": ["X8_ctDNA_Positive", "X3_Age", "X1_Gender"],
        "Name": "Brain Metastasis -> Targeted PFS"
    },
    {
        "Group": "Macroscopic Organ Metastasis",
        "Exposure": "E2_Liver_Met",
        "Outcome": "Y_PFS_IO",
        "MSAS": [],  # 独立根节点，无需调整
        "Name": "Liver Metastasis -> Immunotherapy PFS"
    },
    {
        "Group": "Macroscopic Organ Metastasis",
        "Exposure": "X9_ECOG",
        "Outcome": "Y_OS_TKI_All",
        "MSAS": [],  # 独立根节点，无需调整
        "Name": "ECOG PS (0-1) -> Targeted OS"
    }
]

# ==========================================
# 3. Execute Causal CoxPH Adjustments
# ==========================================
print("\n" + "=" * 80)
print("🛡️ Executing Pearl's Back-door Adjustments for Table 1...")
print("=" * 80)

records = []

for s in scenarios:
    exposure = s["Exposure"]
    outcome = s["Outcome"]
    msas = s["MSAS"]
    event_col = f"{outcome}_Event"

    # 提取需要的列并抛弃空值
    cols_to_keep = [outcome, event_col, exposure] + msas
    analysis_df = df[cols_to_keep].dropna()

    # 拟合带有 L2 惩罚项（防止极值不收敛）的 Cox 因果模型
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(analysis_df, duration_col=outcome, event_col=event_col)

    # 提取所需的统计量
    summary = cph.summary
    hr = summary.loc[exposure, 'exp(coef)']
    lci = summary.loc[exposure, 'exp(coef) lower 95%']
    uci = summary.loc[exposure, 'exp(coef) upper 95%']
    pval = summary.loc[exposure, 'p']

    # 格式化输出文本
    msas_text = ", ".join(msas) if msas else "None (Unconfounded Root Node)"
    hr_ci_text = f"{hr:.3f} ({lci:.3f} - {uci:.3f})"
    p_text = "< 0.001" if pval < 0.001 else f"{pval:.3f}"

    records.append({
        "Causal Pathway": s["Name"],
        "Minimal Sufficient Adjustment Set (MSAS)": msas_text,
        "Causal HR (95% CI)": hr_ci_text,
        "P-value": p_text
    })

    print(f"✅ {s['Name'][:40]:<42} | HR: {hr:.3f} | MSAS: {msas_text}")

# ==========================================
# 4. Export the complete Table 1
# ==========================================
df_table1 = pd.DataFrame(records)
output_path = os.path.join(RESULTS_DIR, "Table1_FULL_Causal_HRs.csv")
df_table1.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print(f"🎉 完整的 Table 1 (包含 6 条核心因果路径) 已成功生成并保存至: {output_path}")
print("=" * 80)