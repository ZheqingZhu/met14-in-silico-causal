import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "./data/nsclc_ctdx_msk_2022/"
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("🔍 Loading MSK 2022 Cohort (N=2621) for 12q13-15 co-amplification analysis...")

# ==========================================================
# 1. Extract Real-World Co-amplification of the 12q13-15 Amplicon
# ==========================================================
df_cna = pd.read_csv(os.path.join(BASE_DIR, 'data_cna.txt'), sep='\t')
target_genes = ['MET', 'MDM2', 'CDK4']

# Set gene names as the index and transpose so patients become rows and genes become columns
df_cna_target = df_cna[df_cna['Hugo_Symbol'].isin(target_genes)].set_index('Hugo_Symbol')
df_cna_t = df_cna_target.T

# In cBioPortal discrete CNA data, '2' denotes Deep Amplification
set_met = set(df_cna_t[df_cna_t['MET'] == 2].index)
set_mdm2 = set(df_cna_t[df_cna_t['MDM2'] == 2].index)
set_cdk4 = set(df_cna_t[df_cna_t['CDK4'] == 2].index)

# ==========================================================
# 2. Generate and Save Venn Diagram - Version A: Annotated (For Manuscript)
# ==========================================================
plt.figure(figsize=(8, 8))
v1 = venn3([set_met, set_mdm2, set_cdk4], ('MET Amp', 'MDM2 Amp', 'CDK4 Amp'),
          set_colors=('#E64B35', '#4DBBD5', '#00A087'), alpha=0.7)
plt.title("Real-World Co-amplification of 12q13-15 Amplicon", fontsize=16, fontweight='bold', pad=20)

output_path_text = os.path.join(RESULTS_DIR, "Fig5a_Validation_12q13-15_Venn_Annotated.pdf")
plt.savefig(output_path_text, format='pdf', bbox_inches='tight')
plt.close()

# ==========================================================
# 3. Generate and Save Venn Diagram - Version B: Clean Vector (For Presentations)
# ==========================================================
plt.figure(figsize=(8, 8))

# Suppress all external set labels
v2 = venn3([set_met, set_mdm2, set_cdk4], set_labels=('', '', ''),
          set_colors=('#E64B35', '#4DBBD5', '#00A087'), alpha=0.7)

# Iterate through and hide all internal intersection subset labels
for text in v2.subset_labels:
    if text is not None:
        text.set_text('')

# Maintain absolute visual purity without any title or text
output_path_notext_pdf = os.path.join(RESULTS_DIR, "Fig5a_Validation_12q13-15_Venn_Clean.pdf")
# Generate a transparent PNG for seamless drag-and-drop into slide decks
output_path_notext_png = os.path.join(RESULTS_DIR, "Fig5a_Validation_12q13-15_Venn_Clean.png")

plt.savefig(output_path_notext_pdf, format='pdf', bbox_inches='tight', transparent=True)
plt.savefig(output_path_notext_png, format='png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# ==========================================================
# 4. 🖨️ Terminal Output of Core Supporting Statistics
# ==========================================================
overlap_all = len(set_met & set_mdm2 & set_cdk4)
overlap_met_mdm2 = len(set_met & set_mdm2)

print("\n" + "=" * 60)
print("📊 [Fig 5a Validation] Real-World 12q13-15 Co-amplification Rates (MSK 2022)")
print("=" * 60)
print(f"Total patients with MET Deep Amplification : {len(set_met)}")
print(f"Total patients with MDM2 Deep Amplification: {len(set_mdm2)}")
print(f"Total patients with CDK4 Deep Amplification: {len(set_cdk4)}")

if len(set_met) > 0:
    print(f"\n👉 Key Finding 1: Among {len(set_met)} MET-amplified patients, {overlap_met_mdm2} also harbored MDM2 amplification.")
    print(f"   (Co-amplification rate: {overlap_met_mdm2 / len(set_met) * 100:.1f}%)")
    print(f"👉 Key Finding 2: Patients harboring deep co-amplification across all three genes (MET+MDM2+CDK4): {overlap_all}")
print("=" * 60)
print(f"✅ Annotated manuscript figure saved to: {output_path_text}")
print(f"✅ Clean presentation figure saved to  : {output_path_notext_png} (Transparent PNG)")