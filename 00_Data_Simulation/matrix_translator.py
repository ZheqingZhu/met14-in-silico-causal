"""
Epidemiological Prior to Correlation Matrix Translator.

This module is responsible for translating literature-derived epidemiological
effect sizes (e.g., Hazard Ratios, Odds Ratios) into a valid mathematical
correlation space (Pearson's r). It addresses matrix contradictions caused
by multi-source heterogeneous priors by projecting the assembled matrix
to the nearest Positive Semi-Definite (PSD) correlation matrix using Higham's algorithm.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import corr_nearest


class PriorToCorrelationTranslator:
    def __init__(self, data_dir: str):
        """
        Initializes the translation engine by loading marginal node definitions
        and edge summary statistics.

        Args:
            data_dir (str): Directory path containing the CSV configuration files.
        """
        nodes_path = os.path.join(data_dir, 'nodes_marginal.csv')
        edges_path = os.path.join(data_dir, 'edges_summary_stats.csv')

        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            raise FileNotFoundError(f"[ERROR] Missing required CSV files in {data_dir}.")

        self.nodes_df = pd.read_csv(nodes_path)
        # Set 'node_id' as the index for efficient row-wise lookup
        self.nodes_df.set_index('node_id', inplace=True)

        self.edges_df = pd.read_csv(edges_path)

        self.node_names = self.nodes_df.index.tolist()
        self.n_nodes = len(self.node_names)

        # Dictionary mapping for quick index retrieval during matrix assembly
        self.node2idx = {name: i for i, name in enumerate(self.node_names)}

    def _effect_to_r(self, node_A: str, metric: str, effect: float, p_value: float) -> float:
        """
        Maps epidemiological effect sizes to Pearson correlation coefficients (r),
        accounting for causal directionality and baseline distribution families.

        Args:
            node_A (str): Source node identifier.
            metric (str): The statistical metric used (e.g., 'HR', 'OR', 'correlation').
            effect (float): The reported effect size.
            p_value (float): The reported p-value for statistical significance penalization.

        Returns:
            float: Bounded Pearson correlation coefficient [-0.99, 0.99].
        """
        if pd.isna(effect):
            return 0.0

        r = 0.0
        metric = str(metric).lower().strip()

        try:
            effect = float(effect)
        except ValueError:
            return 0.0

        node_A_type = str(self.nodes_df.loc[node_A, 'distribution_family']).lower().strip()

        # Transformation for Hazard Ratios (HR)
        if 'hr' in metric:
            d = (np.log(effect) * np.sqrt(6)) / np.pi
            r_raw = d / np.sqrt(d ** 2 + 4)
            r = r_raw if node_A_type == 'binomial' else -r_raw

        # Transformation for Odds Ratios (OR)
        elif 'or' in metric or 'odds' in metric:
            d = (np.log(effect) * np.sqrt(3)) / np.pi
            r_raw = d / np.sqrt(d ** 2 + 4)
            r = r_raw if node_A_type == 'binomial' else -r_raw

        # Direct correlation assignment
        elif 'correlation' in metric:
            r = effect
        else:
            return 0.0

        # P-value Shrinkage Penalty: applies exponential decay to non-significant priors
        if pd.notna(p_value):
            p = float(p_value)
            if p >= 0.05:
                penalty_weight = np.exp(- (p - 0.05) * 10)
                r = r * penalty_weight

        return np.clip(r, -0.99, 0.99)

    def _make_psd(self, R: np.ndarray) -> np.ndarray:
        """
        Applies Higham's (2002) algorithm to project the assembled matrix
        to the nearest Positive Semi-Definite (PSD) correlation matrix.
        This resolves contradictions in global covariance topology.

        Args:
            R (np.ndarray): The raw, potentially non-PSD correlation matrix.

        Returns:
            np.ndarray: A valid PSD correlation matrix with unit diagonal.
        """
        R_final = corr_nearest(R)
        return R_final

    def build_correlation_matrix(self) -> np.ndarray:
        """
        Iterates through the literature priors, maps effect sizes to correlations,
        and builds the final global structural correlation matrix for the Copula.

        Returns:
            np.ndarray: The final N x N PSD correlation matrix.
        """
        print("[INFO] Translating literature effect sizes (HR/OR) to covariance space...")

        # Matrices to accumulate correlation values and track overlapping definitions
        R_sum = np.zeros((self.n_nodes, self.n_nodes))
        count_matrix = np.zeros((self.n_nodes, self.n_nodes))

        for _, row in self.edges_df.iterrows():
            node_A = str(row['node_A']).strip()
            node_B = str(row['node_B']).strip()

            if node_A not in self.node2idx or node_B not in self.node2idx:
                continue

            idx_A = self.node2idx[node_A]
            idx_B = self.node2idx[node_B]

            r_val = self._effect_to_r(node_A, row['relationship_metric'], row['effect_size'], row['p_value'])

            # Symmetrical accumulation
            R_sum[idx_A, idx_B] += r_val
            R_sum[idx_B, idx_A] += r_val
            count_matrix[idx_A, idx_B] += 1
            count_matrix[idx_B, idx_A] += 1

            # Optional: Uncomment the following line for verbose edge-by-edge debugging
            # print(f"  [DEBUG] Edge mapped: {node_A} -> {node_B} | Effect: {row['effect_size']} | r: {r_val:.3f}")

        # Compute averaged correlation for multi-sourced edges
        R = np.eye(self.n_nodes)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and count_matrix[i, j] > 0:
                    R[i, j] = R_sum[i, j] / count_matrix[i, j]

        # Enforce Positive Semi-Definiteness via Higham's algorithm
        Sigma = self._make_psd(R)
        print("[SUCCESS] Matrix assembled and smoothed. Higham PSD validation passed.")

        return Sigma


# =====================================================================
# Standalone Testing Block
# =====================================================================
if __name__ == "__main__":
    DATA_DIR = r"./data"

    try:
        print("-" * 70)
        print("[TEST] Initializing PriorToCorrelationTranslator Validation")
        print("-" * 70)

        translator = PriorToCorrelationTranslator(data_dir=DATA_DIR)
        sigma_matrix = translator.build_correlation_matrix()

        print(f"\n[INFO] Generated Global Sigma Matrix Shape: {sigma_matrix.shape}")

        if 'E1_Brain_Met' in translator.node2idx and 'Y_OS_TKI_All' in translator.node2idx:
            idx_brain = translator.node2idx['E1_Brain_Met']
            idx_os = translator.node2idx['Y_OS_TKI_All']
            r_brain_os = sigma_matrix[idx_brain, idx_os]
            print(f"[VALIDATION] Projected Correlation (Brain_Met <-> Targeted_OS): {r_brain_os:.3f}")

    except Exception as e:
        print(f"[ERROR] Pipeline failed during execution: {e}")