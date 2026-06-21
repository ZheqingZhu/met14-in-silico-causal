"""
Digital Twin Simulation Engine for METex14 Causal Topology Research.

This script acts as the primary executor for the Expectation (E) step of the hybrid 
structural expectation-maximization framework. It translates literature-based marginal
priors and covariance matrices into a high-fidelity synthetic patient cohort using 
Gaussian Copulas, followed by strict marginal recalibration and clinical right-censoring.
"""

import os
import random
import numpy as np
import pandas as pd
from e_step_generator import SyntheticCohortGenerator
from matrix_translator import PriorToCorrelationTranslator


def set_global_determinism(seed: int) -> None:
    """
    Enforces strict deterministic behavior across core computational libraries.
    This ensures bit-for-bit reproducibility of the synthetic digital twin cohorts.

    Args:
        seed (int): The pseudo-random number generator (PRNG) seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[INFO] Global deterministic seed locked to: {seed}")


def run_full_e_step(data_dir: str, n_patients: int = 3000, current_seed: int = 2026) -> pd.DataFrame:
    """
    Executes the Copula-based E-step to generate a synthetic digital twin cohort.

    Args:
        data_dir (str): Directory containing prior distributions and edge matrices.
        n_patients (int): Target number of synthetic patients to generate.
        current_seed (int): Random seed for specific universe cohort generation.

    Returns:
        pd.DataFrame: The generated patient cohort with applied clinical recalibrations.
    """
    print("-" * 70)
    print(f"[INFO] Initializing Copula E-Step Engine | Target Universe Seed: {current_seed}")
    print("-" * 70)

    # 1. Lock computational universe
    set_global_determinism(seed=current_seed)

    # 2. Translate literature priors into covariance structure
    print("[INFO] Building structural correlation matrix from priors...")
    translator = PriorToCorrelationTranslator(data_dir)
    sigma_matrix = translator.build_correlation_matrix()

    # 3. Generate baseline synthetic cohort
    print(f"[INFO] Generating base synthetic cohort (N={n_patients})...")
    generator = SyntheticCohortGenerator(data_dir)
    synthetic_data = generator.generate_cohort(n_patients=n_patients, correlation_matrix=sigma_matrix)

    # 4. Adaptive marginal recalibration for survival endpoints
    print("[INFO] Applying adaptive marginal recalibration for survival endpoints...")
    nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes_marginal.csv')).set_index('node_id')
    survival_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            target_median = float(nodes_df.loc[endpoint, 'base_value'])
            current_median = synthetic_data[endpoint].median()
            calibration_alpha = target_median / current_median
            synthetic_data[endpoint] = synthetic_data[endpoint] * calibration_alpha

    # 5. Right-censoring simulation
    print("[INFO] Simulating clinical right-censoring events...")
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

    # 6. Export results
    output_filename = f"synthetic_cohort_n{n_patients}_seed{current_seed}.csv"
    output_path = os.path.join(data_dir, output_filename)
    synthetic_data.to_csv(output_path, index=False)

    print(f"[SUCCESS] Synthetic cohort generated and saved to: {output_filename}")
    return synthetic_data


if __name__ == "__main__":
    DATA_DIR = r"./data"
    N_PATIENTS = 3000

    # =====================================================================
    # PHASE 1: Generation of the Golden Cohort (Primary Analysis)
    # This specific seed (2026) corresponds to the primary findings reported
    # in the manuscript (e.g., Table 1, Figure 2, Figure 3).
    # =====================================================================
    GOLDEN_SEED = 2026
    print("\n" + "=" * 70)
    print(f"[PHASE 1] Generating Core Digital Twin Cohort (Golden Seed: {GOLDEN_SEED})")
    print("=" * 70)

    df_golden = run_full_e_step(DATA_DIR, n_patients=N_PATIENTS, current_seed=GOLDEN_SEED)

    # Baseline Macroscopic Validation (Sanity Check)
    print(f"\n[VALIDATION] Baseline Macroscopic Penalties Sanity Check (Seed {GOLDEN_SEED})")
    df_events = df_golden[df_golden['Y_PFS_TKI_All_Event'] == 1]

    if 'E1_Brain_Met' in df_events.columns and 'Y_PFS_TKI_All' in df_events.columns:
        med_pfs_no_brain = df_events[df_events['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
        med_pfs_with_brain = df_events[df_events['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()
        print(f"  - Median Targeted PFS (No Brain Met):   {med_pfs_no_brain:.2f} months")
        print(f"  - Median Targeted PFS (With Brain Met): {med_pfs_with_brain:.2f} months")
    print("\n")

    # =====================================================================
    # PHASE 2: Multi-Universe Sensitivity Analysis & Ensemble Averaging
    # =====================================================================
    print("\n" + "=" * 70)
    print("[PHASE 2] Executing Multi-Universe Sensitivity Analysis")
    print("=" * 70)

    robustness_seeds = [2026, 42, 100, 8888, 9999]
    no_brain_pfs_list = []
    with_brain_pfs_list = []

    for seed in robustness_seeds:
        df_test = run_full_e_step(DATA_DIR, n_patients=N_PATIENTS, current_seed=seed)
        df_events_test = df_test[df_test['Y_PFS_TKI_All_Event'] == 1]

        m_no_brain = df_events_test[df_events_test['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
        m_with_brain = df_events_test[df_events_test['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()

        no_brain_pfs_list.append(m_no_brain)
        with_brain_pfs_list.append(m_with_brain)
        print(
            f"  [Verified | Seed {seed:4d}] PFS No Brain Met: {m_no_brain:.2f} mo | PFS With Brain Met: {m_with_brain:.2f} mo")

    # Calculate Mean and SD
    mean_no_brain = np.mean(no_brain_pfs_list)
    sd_no_brain = np.std(no_brain_pfs_list)
    mean_with_brain = np.mean(with_brain_pfs_list)
    sd_with_brain = np.std(with_brain_pfs_list)

    print("-" * 70)
    print("📊 [ENSEMBLE STATISTICS ACROSS 5 UNIVERSES]")
    print(f"  - No Brain Met PFS:   {mean_no_brain:.2f} ± {sd_no_brain:.2f} months")
    print(f"  - With Brain Met PFS: {mean_with_brain:.2f} ± {sd_with_brain:.2f} months")
    print("=" * 70)