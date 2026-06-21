import subprocess
import sys
import argparse
import time


def run_script(script_path, description, demo_mode=False):
    """Execute a single script, capture its output, and print execution status."""
    print(f"\n{'-' * 60}")
    print(f"🚀 [EXECUTING] {description}")
    print(f"📄 {script_path}")
    print(f"{'-' * 60}")

    cmd = [sys.executable, script_path]
    if demo_mode and "01_Network_Discovery" in script_path:
        # Pass the demo flag to computationally intensive network discovery modules
        cmd.append("--demo")

    start_time = time.time()
    try:
        # Run the subprocess and stream output to the console
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ [SUCCESS] {description} completed in {elapsed:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [ERROR] {description} failed! Pipeline aborted.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="METex14 NSCLC Causal Discovery Pipeline Master Script")
    parser.add_argument("--demo", action="store_true", help="Enable fast-track validation mode (reduces optimization iterations in the genetic algorithm module)")
    args = parser.parse_args()

    print("==========================================================")
    print("  METex14 NSCLC Synthetic Cohort & Causal Discovery Pipeline  ")
    print("==========================================================")

    if args.demo:
        print("⚠️  DEMO MODE ENABLED: Fast-track validation active. Skipping global exhaustive search.")

    # 1. Synthetic Cohort Construction
    run_script("00_Data_Simulation/run_digital_twin_simulation.py", "Data Construction (Synthetic Cohort Generation via Copula)")

    # 2. Structural Causal Network Discovery
    run_script("01_Network_Discovery/m_step_ga.py", "Network Discovery (Optimization Iterations)", demo_mode=args.demo)
    run_script("01_Network_Discovery/aggregate_consensus.py", "Network Discovery (Aggregate Consensus)", demo_mode=args.demo)

    # 3. Causal Effect Inference ("do"-calculus)
    run_script("02_Causal_Inference/generate_final_results.py", "Causal Effect Inference (\"do\"-calculus)")
    run_script("02_Causal_Inference/generate_table1_causal_hr.py", "Causal Effect Inference (Generating Table 1)")

    # 4. Counterfactual Survival Visualization
    run_script("03_Visualization/plot_km_counterfactual.py", "Visualization (Counterfactual Survival Curves)")

    # 5. Real-World Topological Validation
    run_script("04_RealWorld_Validation/realworld_validation2018.py", "Validation (ICI Therapy Cohort - Sabari 2018)")
    run_script("04_RealWorld_Validation/realworld_validation2022.py", "Validation (ctDNA Cohort - 2022)")
    run_script("04_RealWorld_Validation/realworld_validation2023.py", "Validation (Macroscopic Metastasis Cohort - 2023)")

    print("\n==========================================================")
    print(" 🎉 Pipeline Execution Complete! All results are in /results.")
    print("==========================================================")


if __name__ == "__main__":
    main()