"""
Causal Inference and G-Computation Engine (Pearl's Do-Calculus).

This module ingests the Golden Digital Twin Cohort and applies Structural
Causal Inference based on the 11-edge Global Consensus DAG. It executes
Cox Proportional Hazards modeling with topological back-door adjustments to
derive unbiased Causal Hazard Ratios (Table 1). Furthermore, it performs
G-computation (Marginal Structural Modeling) to simulate counterfactual
universes (e.g., do(Clearance=1)), extracting absolute median survival times
and landmark probabilities.
"""

import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")


class CausalInferenceEngine:
    def __init__(self, cohort_path: str):
        """
        Initializes the inference engine with the Golden Digital Twin Cohort.
        """
        if not os.path.exists(cohort_path):
            raise FileNotFoundError(f"[ERROR] Golden cohort not found at: {cohort_path}")

        self.df = pd.read_csv(cohort_path)
        print(f"[INFO] Successfully loaded Golden Cohort (N={len(self.df)}).")

    def _get_landmark_survival(self, survival_curve: pd.Series, month: float) -> float:
        """Extracts the survival probability at a specific landmark time (e.g., 12 months)."""
        valid_times = survival_curve.index[survival_curve.index <= month]
        if len(valid_times) == 0:
            return 100.0
        return survival_curve.loc[valid_times[-1]] * 100.0

    def _get_median_survival(self, survival_curve: pd.Series) -> float:
        """Calculates the exact median survival time (crosses 50% probability)."""
        below_median = survival_curve[survival_curve <= 0.5]
        if len(below_median) == 0:
            return np.nan
        return below_median.index[0]

    def estimate_causal_effects(self, exposure: str, outcome: str, backdoor_covariates: list) -> CoxPHFitter:
        """
        Fits a CoxPH model strictly adjusted for DAG-derived backdoor paths.
        """
        event_col = f"{outcome}_Event"
        cols_to_keep = [outcome, event_col, exposure] + backdoor_covariates
        analysis_df = self.df[cols_to_keep].dropna()

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(analysis_df, duration_col=outcome, event_col=event_col)
        return cph

    def run_g_computation(self, cph_model: CoxPHFitter, exposure: str, do_values: list, landmark_mo: float):
        """
        Executes G-computation by forcing the cohort into counterfactual states
        [do(X=x)] and averaging the predicted individual survival curves to obtain
        the unconfounded Marginal Survival Curve.
        """
        results = {}
        for val in do_values:
            # Create a counterfactual universe where EVERYONE receives the intervention 'val'
            df_counterfactual = self.df.copy()
            df_counterfactual[exposure] = val

            # Predict individual survival curves and average them (Marginalization)
            surv_matrix = cph_model.predict_survival_function(df_counterfactual)
            marginal_curve = surv_matrix.mean(axis=1)

            median_surv = self._get_median_survival(marginal_curve)
            landmark_surv = self._get_landmark_survival(marginal_curve, landmark_mo)

            results[f"do({exposure}={val})"] = {
                "Median": median_surv,
                f"{landmark_mo}-mo Rate (%)": landmark_surv
            }
        return results


if __name__ == "__main__":
    # Point to the deterministic Seed 2026 Golden Cohort
    COHORT_PATH = "./data/synthetic_cohort_n3000_seed2026.csv"
    RESULTS_DIR = "./results"

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("-" * 70)
    print("[INFO] Initiating Pearl's Do-Calculus & G-Computation Engine")
    print("-" * 70)

    engine = CausalInferenceEngine(COHORT_PATH)

    # Storage for Table 1 Causal HRs
    table1_records = []

    # =====================================================================
    # Scenario A: Dynamic Molecular Rescue (ctDNA Clearance)
    # DAG identifies Baseline ctDNA+ and Brain Met as independent prognostic factors.
    # =====================================================================
    print("\n[SCENARIO A] Dynamic Rescue: do(ctDNA Clearance)")
    cph_clearance = engine.estimate_causal_effects(
        exposure='M8_ctDNA_Clearance',
        outcome='Y_PFS_TKI_All',
        backdoor_covariates=['X8_ctDNA_Positive', 'E1_Brain_Met']
    )
    g_comp_clearance = engine.run_g_computation(cph_clearance, 'M8_ctDNA_Clearance', [1, 0], landmark_mo=12.0)

    for state, metrics in g_comp_clearance.items():
        print(f"  - {state}: Median PFS = {metrics['Median']:.2f} mo | 12-mo Surv = {metrics['12.0-mo Rate (%)']:.1f}%")

    table1_records.append({
        'Intervention': 'ctDNA Clearance (vs Non-clearance)',
        'Causal HR': cph_clearance.hazard_ratios_['M8_ctDNA_Clearance'],
        'LCI 95%': np.exp(cph_clearance.confidence_intervals_.iloc[0, 0]),
        'UCI 95%': np.exp(cph_clearance.confidence_intervals_.iloc[0, 1]),
        'P-value': cph_clearance.summary.loc['M8_ctDNA_Clearance', 'p']
    })

    # =====================================================================
    # Scenario B: Baseline Penalty (ctDNA Positivity)
    # =====================================================================
    print("\n[SCENARIO B] Baseline Burden: do(Baseline ctDNA Positivity)")
    cph_ctdna_base = engine.estimate_causal_effects(
        exposure='X8_ctDNA_Positive',
        outcome='Y_PFS_TKI_All',
        backdoor_covariates=['E1_Brain_Met', 'X3_Age', 'X1_Gender']  # Exogenous controls
    )
    g_comp_ctdna_base = engine.run_g_computation(cph_ctdna_base, 'X8_ctDNA_Positive', [1, 0], landmark_mo=12.0)

    for state, metrics in g_comp_ctdna_base.items():
        print(f"  - {state}: Median PFS = {metrics['Median']:.2f} mo | 12-mo Surv = {metrics['12.0-mo Rate (%)']:.1f}%")

    # =====================================================================
    # Scenario C: Macroscopic Penalty (Brain Metastasis)
    # =====================================================================
    print("\n[SCENARIO C] Macroscopic Burden: do(Brain Metastasis)")
    cph_brain = engine.estimate_causal_effects(
        exposure='E1_Brain_Met',
        outcome='Y_PFS_TKI_All',
        backdoor_covariates=['X8_ctDNA_Positive', 'X3_Age', 'X1_Gender']
    )
    g_comp_brain = engine.run_g_computation(cph_brain, 'E1_Brain_Met', [1, 0], landmark_mo=12.0)

    for state, metrics in g_comp_brain.items():
        print(f"  - {state}: Median PFS = {metrics['Median']:.2f} mo | 12-mo Surv = {metrics['12.0-mo Rate (%)']:.1f}%")

    # =====================================================================
    # Scenario D: Immunological Long-tail (TMB)
    # =====================================================================
    print("\n[SCENARIO D] Immunological Rescue: do(TMB High vs Low)")
    # Determine 75th and 25th percentiles of TMB
    tmb_high_val = engine.df['X2_TMB'].quantile(0.75)
    tmb_low_val = engine.df['X2_TMB'].quantile(0.25)

    cph_tmb = engine.estimate_causal_effects(
        exposure='X2_TMB',
        outcome='Y_OS_IO',
        backdoor_covariates=['E2_Liver_Met', 'X3_Age']
    )
    g_comp_tmb = engine.run_g_computation(cph_tmb, 'X2_TMB', [tmb_high_val, tmb_low_val], landmark_mo=24.0)

    for state, metrics in g_comp_tmb.items():
        print(
            f"  - do(TMB={state.split('=')[1][:4]}): Median OS = {metrics['Median']:.2f} mo | 24-mo Surv = {metrics['24.0-mo Rate (%)']:.1f}%")

    # =====================================================================
    # Export Table 1
    # =====================================================================
    df_table1 = pd.DataFrame(table1_records)
    df_table1['Causal HR'] = df_table1['Causal HR'].round(3)
    df_table1['95% CI'] = df_table1.apply(lambda r: f"{r['LCI 95%']:.3f} - {r['UCI 95%']:.3f}", axis=1)
    df_table1['P-value'] = df_table1['P-value'].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")

    final_table1 = df_table1[['Intervention', 'Causal HR', '95% CI', 'P-value']]
    output_csv = os.path.join(RESULTS_DIR, "Table1_Causal_HRs.csv")
    final_table1.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("[SUCCESS] Multi-dimensional Do-calculus Execution Complete.")
    print(f"[SUCCESS] Unconfounded Table 1 exported to: {output_csv}")
    print("=" * 70)
