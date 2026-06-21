"""
Global Consensus Aggregation Engine.

This script aggregates the topological discovery results across all parallel digital
twin universes (e.g., 5 distinct random seeds, 100 evolutions each = 500 total runs).
It computes the 'Global Confidence' for each causal edge, strictly filtering out noise
and spurious correlations that fail to replicate across multiple structural states.
The output is the absolute 11-edge Tiered Consensus DAG used for final downstream
G-computation and causal inference.
"""

import os
import glob
import numpy as np
import pandas as pd


def aggregate_universes(data_dir: str):
    """
    Reads all independent tiered edge CSVs, computes global occurrence probabilities,
    and exports the final absolute consensus topology.
    """
    print("-" * 70)
    print("[INFO] Initiating Multi-Universe Topological Aggregation")
    print("-" * 70)

    search_pattern = os.path.join(data_dir, "discovered_dag_tiered_edges_seed*.csv")
    file_list = glob.glob(search_pattern)

    if not file_list:
        raise FileNotFoundError(f"[ERROR] No seed-specific DAG edge files found in {data_dir}.")

    num_universes = len(file_list)
    print(f"[INFO] Detected {num_universes} parallel universe outputs.")

    # 1. Load and combine all evolutionary edges
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        # Assuming each file represents 100 evolutionary runs
        # 'Confidence(%)' represents the occurrence out of 100 runs.
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # 2. Compute Global Confidence (Ensemble Averaging over N_Universes * 100 runs)
    # The total occurrences = mean(Confidence) * occurrence_count
    consensus = combined_df.groupby(['Source', 'Target']).agg(
        universe_appearances=('Confidence(%)', 'count'),
        avg_confidence_in_appeared=('Confidence(%)', 'mean')
    ).reset_index()

    # Calculate absolute probability across ALL simulated runs
    consensus['global_hits'] = (consensus['avg_confidence_in_appeared'] * consensus['universe_appearances'])
    consensus['global_confidence_%'] = (consensus['global_hits'] / num_universes)

    # 3. Apply strict 50% global threshold for putative drivers
    final_edges = consensus[consensus['global_confidence_%'] >= 50.0].copy()
    final_edges = final_edges.sort_values(by='global_confidence_%', ascending=False)

    num_final_edges = len(final_edges)
    print(f"\n[SUCCESS] Extracted {num_final_edges} robust global consensus edges.")

    # 4. Tiering classification based on global confidence
    def assign_tier(conf):
        if conf >= 80.0:
            return "Core Backbone (>=80%)", "Solid"
        else:
            return "Putative Driver (50%-79%)", "Dashed"

    tiers_and_styles = final_edges['global_confidence_%'].apply(assign_tier)
    final_edges['Tier'] = [t[0] for t in tiers_and_styles]
    final_edges['Line_Style'] = [t[1] for t in tiers_and_styles]

    final_export_df = final_edges[['Source', 'Target', 'global_confidence_%', 'Tier', 'Line_Style']].round(2)

    print("\n" + "=" * 70)
    print("[RESULTS] Absolute Universal Consensus Landscape")
    print("=" * 70)
    for _, row in final_export_df.iterrows():
        print(f"  [{row['global_confidence_%']:5.1f}%] {row['Source']} ---> {row['Target']} | {row['Tier']}")

    # 5. Export Final CSVs
    final_edges_path = os.path.join(data_dir, "GLOBAL_CONSENSUS_tiered_edges.csv")
    final_export_df.to_csv(final_edges_path, index=False)

    # 6. Generate Global Core Backbone Adjacency Matrix (>= 80% only) for Pearl's Do-calculus
    # Gather all unique nodes to rebuild matrix
    all_nodes = list(set(combined_df['Source'].unique().tolist() + combined_df['Target'].unique().tolist()))
    adj_matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)

    core_edges = final_export_df[final_export_df['global_confidence_%'] >= 80.0]
    for _, row in core_edges.iterrows():
        adj_matrix.loc[row['Source'], row['Target']] = 1

    final_adj_path = os.path.join(data_dir, "GLOBAL_CONSENSUS_adjacency_matrix.csv")
    adj_matrix.to_csv(final_adj_path)

    print("\n" + "-" * 70)
    print(f"[EXPORT] Final Tiered List saved to:  {final_edges_path}")
    print(f"[EXPORT] Final Core Matrix saved to:  {final_adj_path}")
    print("-" * 70)


if __name__ == "__main__":
    DATA_DIRECTORY = "./data"

    try:
        aggregate_universes(DATA_DIRECTORY)
    except Exception as e:
        print(f"[ERROR] Aggregation failed: {e}")
