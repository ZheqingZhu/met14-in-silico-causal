"""
Structural Maximization (M-step) via Ensemble Evolutionary Search.

This module applies a Genetic Algorithm (GA) combined with Extended Bayesian 
Information Criterion (EBIC) penalization to discover the underlying Directed 
Acyclic Graph (DAG). To ensure absolute robustness and prevent local optima, 
the algorithm runs across multiple parallel computational universes (ensemble 
learning), yielding a Tiered Consensus Causal Landscape.
"""

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures
import statsmodels.api as sm
from lifelines import CoxPHFitter
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

# Initialize DEAP components safely for multiprocessing
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


class DAGEvolutionarySearch:
    def __init__(self, data_path: str):
        """
        Initializes the evolutionary search engine and defines the temporal topology.

        Args:
            data_path (str): Path to the synthetic digital twin cohort CSV.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[ERROR] Cohort data file not found: {data_path}")

        self.df = pd.read_csv(data_path)

        # Node definitions aligned with the digital twin cohort
        self.node_names = [
            'X3_Age', 'X1_Gender', 'X_Smoking', 'X9_ECOG',
            'X6_Histology_PSC', 'X2_TMB', 'X4_Splice_Type',
            'X5_Mutation_Type', 'X7_TP53_Mutation', 'X12_Concurrent_MET_Amp',
            'X10_MDM2_Amp', 'X11_CDK4_Amp',
            'X8_ctDNA_Positive', 'E1_Brain_Met', 'E2_Liver_Met', 'M5_PDL1',
            'X_Line_1L', 'M8_ctDNA_Clearance',
            'Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO'
        ]
        self.n_nodes = len(self.node_names)

        self.node_types = {
            'X3_Age': 'continuous', 'X2_TMB': 'continuous',
            'X1_Gender': 'binary', 'X_Smoking': 'binary', 'X9_ECOG': 'binary',
            'X4_Splice_Type': 'binary', 'X5_Mutation_Type': 'binary',
            'X6_Histology_PSC': 'binary', 'X7_TP53_Mutation': 'binary',
            'X10_MDM2_Amp': 'binary', 'X11_CDK4_Amp': 'binary',
            'X8_ctDNA_Positive': 'binary',
            'E1_Brain_Met': 'binary', 'E2_Liver_Met': 'binary', 'M5_PDL1': 'binary',
            'X12_Concurrent_MET_Amp': 'binary',
            'X_Line_1L': 'binary', 'M8_ctDNA_Clearance': 'binary',
            'Y_PFS_TKI_All': 'survival', 'Y_OS_TKI_All': 'survival',
            'Y_PFS_IO': 'survival', 'Y_OS_IO': 'survival'
        }

        # Enforce temporal tiering to prevent reverse causality
        self.blacklist_matrix = self._build_temporal_tiering_mask()

    def _build_temporal_tiering_mask(self) -> np.ndarray:
        """
        Constructs a strict topological constraint matrix enforcing clinical time-ordering.
        0 denotes a forbidden edge, 1 denotes a permissible edge.
        """
        mat = np.ones((self.n_nodes, self.n_nodes))

        def ban_edge(src: str, dst: str):
            if src in self.node_names and dst in self.node_names:
                idx_src = self.node_names.index(src)
                idx_dst = self.node_names.index(dst)
                mat[idx_src, idx_dst] = 0

        # Tier 1: Baseline Exogenous Variables
        tier1_baseline = ['X3_Age', 'X1_Gender', 'X_Smoking', 'X9_ECOG', 'X6_Histology_PSC',
                          'X2_TMB', 'X4_Splice_Type', 'X5_Mutation_Type', 'X7_TP53_Mutation',
                          'X12_Concurrent_MET_Amp', 'X10_MDM2_Amp', 'X11_CDK4_Amp',
                          'E1_Brain_Met', 'E2_Liver_Met', 'X8_ctDNA_Positive']
        # Tier 2: Baseline Biomarkers
        tier2_biomarker = ['M5_PDL1']
        # Tier 3: Therapeutic Interventions
        tier3_treatment = ['X_Line_1L']
        # Tier 4: Dynamic Responses (Preventing immortal time bias)
        tier4_dynamic = ['M8_ctDNA_Clearance']
        # Tier 5: Survival Endpoints
        tier5_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

        # Rule 1: Strict temporal hierarchy (Higher tiers cannot cause lower tiers)
        tiers = [tier1_baseline, tier2_biomarker, tier3_treatment, tier4_dynamic, tier5_endpoints]
        for i in range(len(tiers)):
            for j in range(i):
                for src in tiers[i]:
                    for dst in tiers[j]:
                        ban_edge(src, dst)

        # Rule 2: Protect 12q13-15 co-amplification cluster topology
        cluster_12q13 = ['X12_Concurrent_MET_Amp', 'X10_MDM2_Amp', 'X11_CDK4_Amp']
        for src in self.node_names:
            if src not in cluster_12q13:
                for dst in cluster_12q13:
                    ban_edge(src, dst)

        # Rule 3: Baseline ctDNA cannot retroactively cause anatomical metastasis
        ban_edge('X8_ctDNA_Positive', 'E1_Brain_Met')
        ban_edge('X8_ctDNA_Positive', 'E2_Liver_Met')

        # Rule 4: Survival endpoints are sink nodes (cannot cause each other)
        for src in tier5_endpoints:
            for dst in tier5_endpoints:
                ban_edge(src, dst)

        # Rule 5: Pure demographic roots (cannot receive incoming edges)
        demographics = ['X3_Age', 'X1_Gender']
        for src in self.node_names:
            for dst in demographics:
                ban_edge(src, dst)

        np.fill_diagonal(mat, 0)
        return mat

    def _remove_cycles(self, adj: np.ndarray) -> np.ndarray:
        """Forces the adjacency matrix to be a Directed Acyclic Graph (DAG)."""
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        try:
            while True:
                cycle = nx.find_cycle(G, orientation='original')
                u, v, _ = cycle[-1]
                G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            pass
        return nx.to_numpy_array(G, nodelist=range(self.n_nodes), dtype=int)

    def enforce_topology(self, individual: list) -> list:
        """Applies hard topological constraints during evolutionary crossover/mutation."""
        adj = np.array(individual).reshape((self.n_nodes, self.n_nodes))
        adj = adj * self.blacklist_matrix
        adj = self._remove_cycles(adj)
        for i, val in enumerate(adj.flatten()):
            individual[i] = val
        return individual

    def cxTwoPoint_enforce(self, ind1: list, ind2: list):
        tools.cxTwoPoint(ind1, ind2)
        self.enforce_topology(ind1)
        self.enforce_topology(ind2)
        return ind1, ind2

    def mutFlipBit_enforce(self, individual: list, indpb: float):
        tools.mutFlipBit(individual, indpb)
        self.enforce_topology(individual)
        return individual,

    def _score_node(self, target_node: str, parent_nodes: list) -> float:
        """
        Calculates the penalized log-likelihood score using EBIC logic.
        Incorporates CoxPH for survival nodes, Logit for binary, OLS for continuous.
        """
        if not parent_nodes: 
            return 0.0

        target_type = self.node_types[target_node]
        X = self.df[parent_nodes]
        y = self.df[target_node]
        k = len(parent_nodes)
        N = len(self.df)
        p = self.n_nodes

        # Extended Bayesian Information Criterion (EBIC) penalization
        gamma = 0.8
        penalty = 0.5 * k * np.log(N) + gamma * k * np.log(p)

        try:
            if target_type == 'survival':
                event_col = f"{target_node}_Event"
                cph_data = self.df[[target_node, event_col] + parent_nodes]
                cph = CoxPHFitter(penalizer=0.1) 
                cph.fit(cph_data, duration_col=target_node, event_col=event_col)
                ll_gain = cph.log_likelihood_ratio_test().test_statistic / 2.0
                return ll_gain - penalty
            elif target_type == 'binary':
                X_with_const = sm.add_constant(X)
                model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=50)
                ll_gain = model.llr / 2.0
                return ll_gain - penalty
            elif target_type == 'continuous':
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                null_model = sm.OLS(y, np.ones(len(y))).fit()
                ll_gain = model.llf - null_model.llf
                return ll_gain - penalty
            else:
                return -10000.0
        except Exception:
            return -10000.0

    def evaluate_dag(self, individual: list) -> tuple:
        """Evaluates the global fitness of a DAG candidate."""
        adj_matrix = np.array(individual).reshape((self.n_nodes, self.n_nodes))
        total_score = 0.0
        for i, target in enumerate(self.node_names):
            parent_indices = np.where(adj_matrix[:, i] == 1)[0]
            if len(parent_indices) > 0:
                parent_names = [self.node_names[idx] for idx in parent_indices]
                total_score += self._score_node(target, parent_names)
            else:
                total_score += self._score_node(target, [])
        return (total_score,)

    def run_evolution(self, pop_size: int = 100, generations: int = 30) -> tuple:
        """Executes the Genetic Algorithm optimization."""
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", lambda: random.choices([0, 1], weights=[0.95, 0.05])[0])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.n_nodes ** 2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate_dag)
        toolbox.register("mate", self.cxTwoPoint_enforce)
        toolbox.register("mutate", self.mutFlipBit_enforce, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        for ind in pop: 
            self.enforce_topology(ind)

        # Suppress stdout during parallel execution to avoid messy console logs
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2,
                                           ngen=generations, verbose=False)

        best_ind = tools.selBest(pop, 1)[0]
        best_matrix = np.array(best_ind).reshape((self.n_nodes, self.n_nodes))
        return best_matrix, self.node_names


# ========================================================
# Parallel Execution Architecture
# ========================================================
def set_global_seed(seed: int):
    """Enforces computational determinism for a specific parallel universe."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_single_universe(args: tuple) -> tuple:
    """Independent process function for ensemble learning."""
    run_idx, data_path, total_runs = args
    current_seed = 42 + run_idx * 100
    set_global_seed(current_seed)

    print(f"[INFO] Initializing Parallel Universe [{run_idx + 1}/{total_runs}] | Seed: {current_seed}")
    searcher = DAGEvolutionarySearch(data_path)
    best_adj, nodes = searcher.run_evolution(pop_size=150, generations=30)
    return best_adj, nodes


# ========================================================
# Main Executor: Multi-Cohort Ensemble Inference
# ========================================================
if __name__ == "__main__":
    # We evaluate the DAG topology across all 5 generated universes
    TARGET_SEEDS = [2026, 42, 100, 8888, 9999]

    # --- Ensemble Parameters ---
    N_RUNS = 100  # 100 independent evolutions PER dataset
    MAX_WORKERS = max(1, os.cpu_count() - 2)

    print("-" * 70)
    print("[INFO] Initiating Multi-Cohort Causal Discovery Pipeline")
    print(f"[INFO] Allocating {MAX_WORKERS} CPU cores for parallel processing.")
    print("-" * 70)

    for dataset_seed in TARGET_SEEDS:
        DATA_PATH = f"../data/synthetic_cohort_n3000_seed{dataset_seed}.csv"
        OUTPUT_ADJ_PATH = f"../data/discovered_dag_adjacency_matrix_seed{dataset_seed}.csv"
        OUTPUT_TIERED_EDGES = f"../data/discovered_dag_tiered_edges_seed{dataset_seed}.csv"

        if not os.path.exists(DATA_PATH):
            print(f"[WARNING] Cohort file not found for seed {dataset_seed}. Skipping...")
            continue

        print(f"\n[INFO] === Processing Cohort Universe: Seed {dataset_seed} ===")
        start_time = time.time()

        # Generate 100 parallel tasks for the current cohort
        tasks = [(i, DATA_PATH, N_RUNS) for i in range(N_RUNS)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(run_single_universe, tasks))

        end_time = time.time()
        print(
            f"\n[SUCCESS] Cohort {dataset_seed}: {N_RUNS} searches completed in {(end_time - start_time) / 60:.2f} mins.")

        # --- Evaluate Tiered Consensus for Current Cohort ---
        nodes = results[0][1]
        n_nodes = len(nodes)
        consensus_matrix_sum = np.zeros((n_nodes, n_nodes))

        for best_adj, _ in results:
            consensus_matrix_sum += best_adj

        edge_confidence = consensus_matrix_sum / N_RUNS
        edges_data = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                conf = edge_confidence[i, j]
                if conf >= 0.50:
                    source = nodes[i]
                    target = nodes[j]
                    conf_pct = conf * 100

                    if conf >= 0.80:
                        tier = "Core Backbone (>=80%)"
                        line_style = "Solid"
                    else:
                        tier = "Putative Driver (50%-79%)"
                        line_style = "Dashed"

                    edges_data.append({
                        "Source": source,
                        "Target": target,
                        "Confidence(%)": conf_pct,
                        "Tier": tier,
                        "Line_Style": line_style
                    })

        if not edges_data:
            print(f"[WARNING] No causal edges > 50% found for Cohort {dataset_seed}.")
        else:
            df_edges = pd.DataFrame(edges_data)
            df_edges = df_edges.sort_values(by="Confidence(%)", ascending=False)
            df_edges.to_csv(OUTPUT_TIERED_EDGES, index=False)

            final_core_adj = np.where(edge_confidence >= 0.80, 1, 0)
            df_adj = pd.DataFrame(final_core_adj, index=nodes, columns=nodes)
            df_adj.to_csv(OUTPUT_ADJ_PATH)

            print(f"[SUCCESS] Exported Tiered Edges: {OUTPUT_TIERED_EDGES}")
            print(f"[SUCCESS] Exported Core Matrix:  {OUTPUT_ADJ_PATH}")
            print("-" * 70)