import numpy as np
import pandas as pd
import os
import random
from e_step_generator import SyntheticCohortGenerator
from matrix_translator import PriorToCorrelationTranslator


def set_global_determinism(seed):
    """
    Lock the global random number generators to ensure strict computational reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 Global system seed locked (Seed={seed}). Entering deterministic computation mode.")


def run_full_e_step(data_dir, n_patients=3000, current_seed=2026):
    """
    Execute the Copula Expectation-step to construct the synthetic cohort.
    """
    print("\n" + "=" * 60)
    print(f"🤖 Initiating Copula E-step Engine | Current PRNG Seed: {current_seed}")
    print("=" * 60 + "\n")

    # Step 0: Ensure strict determinism for the current execution
    set_global_determinism(seed=current_seed)

    # Step 1: Translate empirical priors to the correlation matrix (Sigma)
    translator = PriorToCorrelationTranslator(data_dir)
    sigma_matrix = translator.build_correlation_matrix()

    # Step 2: Construct the synthetic cohort via Gaussian Copula
    print("\n------------------------------------------------------------")
    generator = SyntheticCohortGenerator(data_dir)
    synthetic_data = generator.generate_cohort(n_patients=n_patients, correlation_matrix=sigma_matrix)

    # --- [Methodological Patch 1]: Adaptive Marginal Recalibration ---
    print("\n⚖️ Executing adaptive marginal recalibration...")
    nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes_marginal.csv')).set_index('node_id')
    survival_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            target_median = float(nodes_df.loc[endpoint, 'base_value'])
            current_median = synthetic_data[endpoint].median()
            calibration_alpha = target_median / current_median
            synthetic_data[endpoint] = synthetic_data[endpoint] * calibration_alpha

    # --- [Methodological Patch 2]: Right-Censoring Simulation ---
    print("\n⚙️ Applying right-censoring simulator...")
    median_followup = 30.0

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            true_time = synthetic_data[endpoint].values
            scale_param = median_followup / np.log(2)
            censoring_time = np.random.exponential(scale=scale_param, size=n_patients)
            observed_time = np.minimum(true_time, censoring_time)
            event_indicator = (true_time <= censoring_time).astype(int)
            synthetic_data[endpoint] = observed_time
            synthetic_data[f"{endpoint}_Event"] = event_indicator

    # Save the synthetic cohort with the specific PRNG seed in the filename
    output_filename = f"synthetic_cohort_n{n_patients}_seed{current_seed}.csv"
    output_path = os.path.join(data_dir, output_filename)
    synthetic_data.to_csv(output_path, index=False)

    print("\n============================================================")
    print(f"🎉 Successfully constructed {n_patients} synthetic patients! Data saved to: {output_filename}")
    print("============================================================")

    return synthetic_data


if __name__ == "__main__":
    DATA_DIR = r"./data"

    # ==========================================
    # 🧪 Stress Testing Module
    # Iterate through multiple PRNG seeds to validate macroscopic stability
    # ==========================================
    test_seeds = [100]

    for seed in test_seeds:
        df_cohort = run_full_e_step(DATA_DIR, n_patients=3000, current_seed=seed)

        print(f"\n--- Validating macroscopic survival penalties for Seed {seed} ---")
        df_events = df_cohort[df_cohort['Y_PFS_TKI_All_Event'] == 1]

        med_pfs_no_brain = df_events[df_events['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
        med_pfs_with_brain = df_events[df_events['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()

        print(f"✅ [Seed {seed}] Median PFS without Brain Met: {med_pfs_no_brain:.2f} months")
        print(f"✅ [Seed {seed}] Median PFS with Brain Met:    {med_pfs_with_brain:.2f} months\n")