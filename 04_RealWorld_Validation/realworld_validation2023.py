import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "./data/luad_mskcc_2023_met_organotropism/"
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("🔍 Loading MSK 2023 Organotropism (Cancer Cell) real-world clinical cohort...")

# ==========================================================
# 1. Load Clinical Data
# ==========================================================
df_patient = pd.read_csv(os.path.join(BASE_DIR, 'data_clinical_patient.txt'), sep='\t', skiprows=4)

# ==========================================================
# 2. Clean Survival Data (Overall Survival - OS)
# ==========================================================
df_patient['Event'] = df_patient['OS_STATUS'].apply(lambda x: 1 if 'DECEASED' in str(x).upper() else 0)
df_patient['OS_MONTHS'] = pd.to_numeric(df_patient['OS_MONTHS'], errors='coerce')
df_patient = df_patient.dropna(subset=['OS_MONTHS', 'Event'])
df_patient = df_patient[df_patient['OS_MONTHS'] > 0]

# ==========================================================
# 3. [Precise Matching] Parse '1:Recurrent' and '0:RecurrenceFree'
# ==========================================================
def parse_status(val):
    text = str(val).strip().upper()
    # If the text contains '1:' or 'RECURRENT', confirm organ metastasis
    if '1:RECURRENT' in text or '1:' in text:
        return 1
    # Impute '0:RecurrenceFree' or NaN (missing data) to the control group
    else:
        return 0

df_patient['Brain_Met'] = df_patient['CNS_STATUS'].apply(parse_status)
df_patient['Liver_Met'] = df_patient['LIVER_STATUS'].apply(parse_status)

print(f"\n📊 [Final Cohort] Stratification complete (Total valid follow-ups: {len(df_patient)}):")
print(f"  🧠 Confirmed Brain Met (CNS) patients: {df_patient['Brain_Met'].sum()}")
print(f"  🩸 Confirmed Liver Met patients      : {df_patient['Liver_Met'].sum()}")


# ==========================================================
# 4. Generate Annotated Real-World KM Validation Curves
# ==========================================================
def plot_real_world_km(ax, condition_col, title, label_1, label_0, color_1, color_0):
    kmf_1 = KaplanMeierFitter()
    kmf_0 = KaplanMeierFitter()

    mask_1 = df_patient[condition_col] == 1
    mask_0 = df_patient[condition_col] == 0

    if mask_1.sum() == 0 or mask_0.sum() == 0:
        ax.set_title(f"{title} (Data Missing)", fontsize=12)
        return float('nan'), float('nan'), float('nan')

    kmf_1.fit(df_patient[mask_1]['OS_MONTHS'], df_patient[mask_1]['Event'], label=label_1)
    kmf_0.fit(df_patient[mask_0]['OS_MONTHS'], df_patient[mask_0]['Event'], label=label_0)

    med_1 = kmf_1.median_survival_time_
    med_0 = kmf_0.median_survival_time_

    # Log-rank test
    results = logrank_test(df_patient[mask_1]['OS_MONTHS'], df_patient[mask_0]['OS_MONTHS'],
                           event_observed_A=df_patient[mask_1]['Event'], event_observed_B=df_patient[mask_0]['Event'])
    p_val = results.p_value
    p_str = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.4f}"

    kmf_1.plot_survival_function(ax=ax, color=color_1, linewidth=2.5, ci_show=False)
    kmf_0.plot_survival_function(ax=ax, color=color_0, linewidth=2.5, ci_show=False)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Overall Survival (Months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xlim(0, 100) # Slightly widen X-axis for comprehensive view

    # Handle cases where median survival cannot be calculated
    med_1_str = f"{med_1:.1f}" if not np.isinf(med_1) else ">100"
    med_0_str = f"{med_0:.1f}" if not np.isinf(med_0) else ">100"

    legend_text = f"{label_1} (Med: {med_1_str} mo)\n{label_0} (Med: {med_0_str} mo)\n{p_str}"
    ax.text(0.5, 0.75, legend_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.get_legend().remove()

    return med_1, med_0, p_val

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

print("\n📈 Executing Log-rank survival analysis...")
m1_b, m0_b, p_b = plot_real_world_km(axes[0], 'Brain_Met', "Real-World OS: Brain Metastasis Penalty",
                                     "Brain Met (+)", "Brain Met (-)", '#E64B35', '#3C5488')
m1_l, m0_l, p_l = plot_real_world_km(axes[1], 'Liver_Met', "Real-World OS: Liver Metastasis Penalty",
                                     "Liver Met (+)", "Liver Met (-)", '#00A087', '#3C5488')

plt.suptitle("Real-World Validation of Macroscopic Survival Penalties (Cancer Cell, 2023 Cohort)",
             fontsize=16, fontweight='bold', y=1.05)

output_path_annotated = os.path.join(RESULTS_DIR, "Validation_Macroscopic_Penalty_Annotated.pdf")
plt.savefig(output_path_annotated, format='pdf', bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "=" * 60)
print("🎯 [Validation Data] Real-world survival penalties for macroscopic organotropism")
print("=" * 60)
print(f"Brain Met (+) Median OS: {m1_b:>5.2f} mo vs Brain Met (-) Median OS: {m0_b:>5.2f} mo")
print(f"Liver Met (+) Median OS: {m1_l:>5.2f} mo vs Liver Met (-) Median OS: {m0_l:>5.2f} mo")
print(f"✅ Annotated real-world survival curves saved to: {output_path_annotated}")
print("=" * 60)

# ==========================================================
# 5. Extract Real-World Hazard Ratios via Cox Proportional Hazards
# ==========================================================
def get_hr(condition_col):
    cph = CoxPHFitter()
    df_cox = df_patient[[condition_col, 'OS_MONTHS', 'Event']].dropna()
    cph.fit(df_cox, duration_col='OS_MONTHS', event_col='Event')
    hr = cph.hazard_ratios_.iloc[0]
    ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
    ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
    return hr, ci_lower, ci_upper

hr_b, ci_l_b, ci_u_b = get_hr('Brain_Met')
hr_l, ci_l_l, ci_u_l = get_hr('Liver_Met')

print(f"🧠 Brain Met Real-World HR = {hr_b:.2f} (95% CI: {ci_l_b:.2f} - {ci_u_b:.2f})")
print(f"🩸 Liver Met Real-World HR = {hr_l:.2f} (95% CI: {ci_l_l:.2f} - {ci_u_l:.2f})")


# ==========================================================
# 🎨 6. Global NPG-style Minimalist Plotting Configuration
# ==========================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "xtick.labelsize": 10,   # Retain numeric ticks and slightly enlarge for presentation visibility
    "ytick.labelsize": 10,
    "axes.linewidth": 1.5,   # Slightly thicken axes for a publication-grade aesthetic
    "lines.linewidth": 2.5,  # Thicken survival curves for stronger visual impact
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# ==========================================================
# 📊 7. Generate Pure KM Curves (No legends, no text labels, numeric ticks only)
# ==========================================================
def plot_real_world_km_ppt_template(ax, condition_col, color_1, color_0):
    kmf_1 = KaplanMeierFitter()
    kmf_0 = KaplanMeierFitter()

    mask_1 = df_patient[condition_col] == 1
    mask_0 = df_patient[condition_col] == 0

    if mask_1.sum() == 0 or mask_0.sum() == 0:
        return

    # Fit data
    kmf_1.fit(df_patient[mask_1]['OS_MONTHS'], df_patient[mask_1]['Event'])
    kmf_0.fit(df_patient[mask_0]['OS_MONTHS'], df_patient[mask_0]['Event'])

    # Plot curves (disable built-in confidence intervals to maintain minimalist purity)
    kmf_1.plot_survival_function(ax=ax, color=color_1, ci_show=False)
    kmf_0.plot_survival_function(ax=ax, color=color_0, ci_show=False)

    # Draw 50% survival probability horizontal reference line
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, zorder=1)

    # Draw vertical median survival reference line (if applicable)
    med_1 = kmf_1.median_survival_time_
    med_0 = kmf_0.median_survival_time_
    if not np.isinf(med_1):
        ax.plot([med_1, med_1], [0, 0.5], color=color_1, linestyle='--', alpha=0.7, zorder=2)
    if not np.isinf(med_0):
        ax.plot([med_0, med_0], [0, 0.5], color=color_0, linestyle='--', alpha=0.7, zorder=2)

    # ---------------------------------------------------------
    # [Core Modification] Strip titles, axis labels, and legends
    # ---------------------------------------------------------
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)  # Standardize X-axis to 120 months (10 years) for panel alignment
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove legend
    if ax.get_legend():
        ax.get_legend().remove()

# ==========================================================
# 💾 8. Construct 1x2 Panel and Export Transparent Vector Plot
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plt.subplots_adjust(wspace=0.3)

# Apply NPG color palette: Intervention/Danger (Red), Control/Safe (Blue)
C_DANGER = '#E64B35'
C_SAFE = '#4DBBD5'

# Panel C (Left): Brain Met Validation
plot_real_world_km_ppt_template(axes[0], 'Brain_Met', color_1=C_DANGER, color_0=C_SAFE)

# Panel D (Right): Liver Met Validation
plot_real_world_km_ppt_template(axes[1], 'Liver_Met', color_1=C_DANGER, color_0=C_SAFE)

# Export clean PDF with transparent background
output_path_clean = os.path.join(RESULTS_DIR, "Fig5_CD_Macroscopic_Validation_Clean.pdf")
plt.savefig(output_path_clean, format='pdf', bbox_inches='tight', transparent=True)
plt.close()

print("\n" + "=" * 60)
print(f"✅ [Presentation Vector] NPG-style, text-free, transparent background vector plots generated!")
print(f"📁 Path: {output_path_clean}")
print("=" * 60)
print("💡 Tip: Import this PDF into your slide deck and manually overlay text boxes (e.g., HR, medians) to guarantee absolute font consistency across your presentation.")