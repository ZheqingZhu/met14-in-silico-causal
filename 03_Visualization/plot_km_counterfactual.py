import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# 📂 1. Global Minimalist Configuration (Vector Base Template)
# ==========================================================
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

C_INTERVENTION = "#E64B35"  # NPG Red
C_CONTROL = "#4DBBD5"       # NPG Blue

DATA_PATH = "./data/synthetic_cohort_n3000_seed2026.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"[ERROR] Synthetic cohort data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)


# ==========================================================
# ⚙️ 2. Core Causal Inference & Counterfactual Plotting Engine
# ==========================================================
def plot_and_extract_causal(ax, exposure, adjustment_set, duration, event,
                            do_values, labels, landmark_time, x_max):
    """
    Executes Pearl's Do-calculus (via G-computation based on a Cox model 
    adjusted by the Minimal Sufficient Adjustment Set) and generates a pure 
    base plot without text labels.
    """
    # Step 1: Fit Causal Cox Model (Strictly applying DAG back-door adjustment)
    cph = CoxPHFitter(penalizer=0.01)
    covariates = [exposure] + adjustment_set
    valid_df = df[[duration, event] + covariates].dropna()
    cph.fit(valid_df, duration_col=duration, event_col=event)

    # Step 2: Simulate Parallel Universes (G-computation)
    df_do1 = valid_df.copy()
    df_do1[exposure] = do_values[0]
    df_do0 = valid_df.copy()
    df_do0[exposure] = do_values[1]

    sf_1 = cph.predict_survival_function(df_do1).mean(axis=1)
    sf_0 = cph.predict_survival_function(df_do0).mean(axis=1)

    sf_1_plot = sf_1[sf_1.index <= x_max]
    sf_0_plot = sf_0[sf_0.index <= x_max]

    # Step 3: Extract Median and Landmark Survival Data for Console Output
    med_1 = sf_1[sf_1 <= 0.5].index[0] if not sf_1[sf_1 <= 0.5].empty else np.nan
    med_0 = sf_0[sf_0 <= 0.5].index[0] if not sf_0[sf_0 <= 0.5].empty else np.nan

    idx_1 = min(sf_1.index, key=lambda x: abs(x - landmark_time))
    idx_0 = min(sf_0.index, key=lambda x: abs(x - landmark_time))
    rate_1 = sf_1[idx_1] * 100
    rate_0 = sf_0[idx_0] * 100

    print(f"  ▶ {labels[0]:<17}: Median = {med_1:>5.2f} mo | {landmark_time}-mo Survival = {rate_1:.1f}%")
    print(f"  ▶ {labels[1]:<17}: Median = {med_0:>5.2f} mo | {landmark_time}-mo Survival = {rate_0:.1f}%")

    # Step 4: Draw Counterfactual Curves & Causal Effect Area (Shading)
    ax.plot(sf_1_plot.index, sf_1_plot.values, color=C_INTERVENTION, zorder=4)
    ax.plot(sf_0_plot.index, sf_0_plot.values, color=C_CONTROL, zorder=4, linestyle='--')
    ax.fill_between(sf_1_plot.index, sf_0_plot.values, sf_1_plot.values, color='#B0BEC5', alpha=0.3, zorder=1)

    # Step 5: Draw Reference Lines
    ax.axhline(0.5, color='gray', linestyle='-.', alpha=0.5, zorder=2)
    if not np.isnan(med_1):
        ax.plot([med_1, med_1], [0, 0.5], color=C_INTERVENTION, linestyle=':', alpha=0.8, zorder=3)
    if not np.isnan(med_0):
        ax.plot([med_0, med_0], [0, 0.5], color=C_CONTROL, linestyle=':', alpha=0.8, zorder=3)
    if landmark_time:
        ax.axvline(landmark_time, color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Step 6: Vector Graphic Layout (Retain numeric ticks, strip text labels)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0, x_max])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


# ==========================================================
# 📊 3. Master Execution: Applying MSAS from Table 1
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏆 Figure 3: Core Causal Value Extractor & Base Plot Generator")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    print("\n[Panel A] Dynamic Rescue: ctDNA Clearance")
    # MSAS: Baseline ctDNA(+), Brain Metastasis
    plot_and_extract_causal(axes[0, 0], 'M8_ctDNA_Clearance', ['X8_ctDNA_Positive', 'E1_Brain_Met'],
                            'Y_PFS_TKI_All', 'Y_PFS_TKI_All_Event',
                            do_values=(1, 0), labels=("Cleared (1)", "Not Cleared (0)"), landmark_time=12, x_max=36)

    print("\n[Panel B] Molecular Burden: Baseline ctDNA+")
    # MSAS: Brain Metastasis, Age, Sex
    plot_and_extract_causal(axes[0, 1], 'X8_ctDNA_Positive', ['E1_Brain_Met', 'X3_Age', 'X1_Gender'],
                            'Y_PFS_TKI_All', 'Y_PFS_TKI_All_Event',
                            do_values=(0, 1), labels=("Negative (0)", "Positive (1)"), landmark_time=12, x_max=36)

    print("\n[Panel C] Macroscopic Burden: Baseline Brain Metastasis")
    # MSAS: Baseline ctDNA(+), Age, Sex
    plot_and_extract_causal(axes[1, 0], 'E1_Brain_Met', ['X8_ctDNA_Positive', 'X3_Age', 'X1_Gender'],
                            'Y_PFS_TKI_All', 'Y_PFS_TKI_All_Event',
                            do_values=(0, 1), labels=("No Brain Met (0)", "Brain Met (1)"), landmark_time=12, x_max=36)

    print("\n[Panel D] ICI Long-Tail: TMB Disparity")
    # MSAS: Liver Metastasis, Age
    tmb_75 = df['X2_TMB'].quantile(0.75)
    tmb_25 = df['X2_TMB'].quantile(0.25)
    plot_and_extract_causal(axes[1, 1], 'X2_TMB', ['E2_Liver_Met', 'X3_Age'],
                            'Y_OS_IO', 'Y_OS_IO_Event',
                            do_values=(tmb_75, tmb_25), labels=("High TMB (P75)", "Low TMB (P25)"), landmark_time=24,
                            x_max=60)

    # Output Vector Base Template
    output_filename = os.path.join(RESULTS_DIR, "Figure_3_Counterfactual_Vector_Template.pdf")
    plt.savefig(output_filename, format='pdf', transparent=True)

    print("\n" + "=" * 60)
    print(f"✅ Vector base plots (with numeric ticks, transparent background) saved to: {output_filename}")
    print("=" * 60)
