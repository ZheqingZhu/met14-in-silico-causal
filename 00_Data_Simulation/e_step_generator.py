"""
Gaussian Copula-based Synthetic Cohort Generator.

This module applies the Probability Integral Transform (PIT) within a Gaussian Copula
framework. It maps multivariate normal latent variables (structurally constrained by
the translated covariance matrix) into realistic clinical distributions (e.g., Binomial,
Log-normal, Weibull) to synthesize high-fidelity digital twin cohorts.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


class SyntheticCohortGenerator:
    def __init__(self, data_dir: str):
        """
        Initializes the generator by loading marginal distribution priors.

        Args:
            data_dir (str): Directory containing the configuration CSV files.
        """
        self.nodes_csv_path = os.path.join(data_dir, 'nodes_marginal.csv')
        self.edges_csv_path = os.path.join(data_dir, 'edges_summary_stats.csv')

        try:
            self.nodes_df = pd.read_csv(self.nodes_csv_path)
            self.nodes_df.set_index('node_id', inplace=True)
            self.node_names = self.nodes_df.index.tolist()
            self.num_nodes = len(self.node_names)
            print(f"[INFO] Successfully loaded marginal definitions for {self.num_nodes} nodes.")
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Missing marginal configuration file at: {self.nodes_csv_path}")

    def _parse_marginal(self, node_name: str, u_array: np.ndarray) -> np.ndarray:
        """
        Probability Integral Transform (PIT) Engine.
        Inverts uniform variables U(0,1) into target clinical probability distributions.

        Args:
            node_name (str): Identifier of the clinical variable.
            u_array (np.ndarray): Array of uniformly distributed latent variables.

        Returns:
            np.ndarray: Clinically mapped physiological or survival variables.
        """
        row = self.nodes_df.loc[node_name]
        dist_family = str(row['distribution_family']).lower().strip()
        base_val = float(row['base_value'])

        # 1. Binomial Distribution (Binary traits: mutations, brain metastasis, PD-L1 status)
        if dist_family == 'binomial':
            return (u_array < base_val).astype(int)

        # 2. Normal Distribution (Continuous symmetric variables e.g., age)
        elif dist_family == 'normal':
            std_dev = base_val * 0.1
            return stats.norm.ppf(u_array, loc=base_val, scale=std_dev)

        # 2.5 Truncated Normal Distribution (Continuous variables with strict biological bounds)
        elif dist_family == 'truncated_normal':
            lower = float(row['lower_bound'])
            upper = float(row['upper_bound'])
            # Estimate standard deviation assuming range spans roughly 4 standard deviations
            std_dev = (upper - lower) / 4.0
            a = (lower - base_val) / std_dev
            b = (upper - base_val) / std_dev
            return stats.truncnorm.ppf(u_array, a, b, loc=base_val, scale=std_dev)

        # 3. Log-normal Distribution (Right-skewed biomarkers e.g., TMB)
        elif dist_family == 'lognormal':
            mu = np.log(base_val)
            sigma = 0.5  # Assumed reasonable variance coefficient for genomic burdens
            return stats.lognorm.ppf(u_array, s=sigma, scale=np.exp(mu))

        # 4. Weibull Distribution (Baseline absolute survival times OS/PFS)
        elif dist_family == 'weibull':
            shape_k = 1.2  # Oncology survival shape parameter typically ranges 1.0 - 1.5
            # Reverse-engineer scale parameter (lambda) from target median
            scale_lambda = base_val / (np.log(2) ** (1 / shape_k))
            return stats.weibull_min.ppf(u_array, c=shape_k, scale=scale_lambda)

        else:
            raise ValueError(f"[ERROR] Unknown distribution family: {dist_family} for node {node_name}")

    def generate_cohort(self, n_patients: int = 1000, correlation_matrix: np.ndarray = None) -> pd.DataFrame:
        """
        Synthesizes the digital twin cohort utilizing the Gaussian Copula mechanism.

        Args:
            n_patients (int): Target number of synthetic patients.
            correlation_matrix (np.ndarray): The structural PSD correlation matrix.

        Returns:
            pd.DataFrame: The synthetic patient cohort with absolute un-censored survival times.
        """
        print(f"[INFO] Initializing Gaussian Copula generation for N={n_patients} synthetic patients...")

        # Default to identity matrix (independent assumption) if no structure is provided
        if correlation_matrix is None:
            correlation_matrix = np.eye(self.num_nodes)

        # Step 1: Sample from Multivariate Normal distribution (Latent Z space)
        mu = np.zeros(self.num_nodes)
        Z = np.random.multivariate_normal(mu, correlation_matrix, size=n_patients)

        # Step 2: Transform Z into Uniform distribution U(0,1) via Normal CDF
        U = stats.norm.cdf(Z)

        # Step 3: Map U into clinical space via inverse CDF (PIT)
        synthetic_data = {}
        for i, node in enumerate(self.node_names):
            synthetic_data[node] = self._parse_marginal(node, U[:, i])

        df_cohort = pd.DataFrame(synthetic_data)

        print("[SUCCESS] Base digital twin cohort formulated (Absolute times, pending external right-censoring).")
        return df_cohort
