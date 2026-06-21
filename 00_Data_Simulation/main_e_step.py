import numpy as np
import pandas as pd
import os
import random
from e_step_generator import SyntheticCohortGenerator
from matrix_translator import PriorToCorrelationTranslator


def set_global_determinism(seed=42):
    """
    [Methodological Patch 3]: Absolute Global Determinism Lock.
    Locks randomness across the OS, Python interpreter, and scientific compute
    libraries to ensure 100% bitwise reproducibility of the synthetic cohort.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Global system seed locked (Seed={seed}). Entering Deterministic Mode.")


def run_full_e_step(data_dir, n_patients=3000):
    """
    Execute the Copula Expectation-step to construct the synthetic cohort.
    """
    print("============================================================")
    print("🤖 Initiating Copula Expectation-Maximization (E-step) Engine")
    print("============================================================\n")

    # [Core Fix]: Enforce strict determinism before any computation begins
    set_global_determinism(seed=2026)

    # Step 1: Translate empirical literature priors into a Sigma correlation matrix
    translator = PriorToCorrelationTranslator(data_dir)
    sigma_matrix = translator.build_correlation_matrix()

    # Step 2: Feed the Sigma matrix to the generator to construct the synthetic cohort
    print("\n------------------------------------------------------------")
    generator = SyntheticCohortGenerator(data_dir)
    synthetic_data = generator.generate_cohort(n_patients=n_patients, correlation_matrix=sigma_matrix)

    # ---------------------------------------------------------
    # [Methodological Patch 1]: Adaptive Marginal Recalibration
    # Eliminates macroscopic survival drift caused by the enrichment
    # of high-risk factors (e.g., Brain Met, ctDNA+).
    # ---------------------------------------------------------
    print("\n⚖️ Executing adaptive marginal recalibration (anchoring to literature baselines)...")
    nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes_marginal.csv')).set_index('node_id')
    survival_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            target_median = float(nodes_df.loc[endpoint, 'base_value'])
            current_median = synthetic_data[endpoint].median()
            calibration_alpha = target_median / current_median
            synthetic_data[endpoint] = synthetic_data[endpoint] * calibration_alpha
            print(
                f"  🔧 [{endpoint}] Target: {target_median:.1f} | Drifted Actual: {current_median:.2f} | Correction Alpha: {calibration_alpha:.3f}")

    # ---------------------------------------------------------
    # [Methodological Patch 2]: Right-Censoring Simulation
    # Generates realistic follow-up and drop-out distributions.
    # ---------------------------------------------------------
    print("\n⚙️ Applying right-censoring simulator...")
    median_followup = 30.0

    # Note: No need to re-seed here; the global lock is actively enforced.
    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            true_time = synthetic_data[endpoint].values
            scale_param = median_followup / np.log(2)
            censoring_time = np.random.exponential(scale=scale_param, size=n_patients)
            observed_time = np.minimum(true_time, censoring_time)
            event_indicator = (true_time <= censoring_time).astype(int)
            synthetic_data[endpoint] = observed_time
            synthetic_data[f"{endpoint}_Event"] = event_indicator
            event_rate = event_indicator.mean() * 100
            print(
                f"  ✅ [{endpoint}] Censoring applied: Mean Follow-up = {observed_time.mean():.1f} mo, Event Rate = {event_rate:.1f}%")

    # Step 3: Save results for downstream structural discovery
    output_path = os.path.join(data_dir, f"synthetic_cohort_n{n_patients}.csv")
    synthetic_data.to_csv(output_path, index=False)

    print("\n============================================================")
    print(f"🎉 Successfully generated {n_patients} synthetic patients with causal constraints and realistic censoring!")
    print(f"📁 Data saved to: {output_path}")
    print("============================================================")

    return synthetic_data


if __name__ == "__main__":
    DATA_DIR = r"./data"
    df_cohort = run_full_e_step(DATA_DIR, n_patients=3000)

    print("\n--- Clinical Law Cross-Validation (Survival penalty of Brain Metastasis) ---")
    df_events = df_cohort[df_cohort['Y_PFS_TKI_All_Event'] == 1]
    med_pfs_no_brain = df_events[df_events['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
    med_pfs_with_brain = df_events[df_events['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()

    print(f"✅ Median PFS without Brain Met: {med_pfs_no_brain:.2f} months")
    print(f"✅ Median PFS with Brain Met:    {med_pfs_with_brain:.2f} months")

    if med_pfs_with_brain < med_pfs_no_brain:
        print(
            "💡 Conclusion: Perfect alignment! By injecting the negative correlation prior, the model autonomously applies the expected macroscopic survival penalty to patients with brain metastasis.")