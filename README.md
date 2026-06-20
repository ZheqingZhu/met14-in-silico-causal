# Inferring the Causal Architecture of METex14 NSCLC via a Synthetic Cohort

[![Python 3.10.4](https://img.shields.io/badge/python-3.10.4-blue.svg)](https://www.python.org/downloads/release/python-3104/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This repository contains the official code and analytical pipeline for our study investigating the causal architecture and therapeutic mediators of *MET*ex14 non-small cell lung cancer (NSCLC). 

Evaluating rare mutational subsets is fundamentally hindered by data fragmentation and multidimensional confounding. Designed with the rigorous standards of computer science and medical-engineering research, this repository provides a fully reproducible computational framework. We utilize a multi-family Gaussian copula sampling framework to construct a high-fidelity 3,000-patient synthetic cohort, followed by an ensemble score-based causal discovery algorithm (regularized by EBIC) and Pearl's *do*-calculus to isolate absolute survival effects from baseline confounding.

## 🏗️ Repository Architecture

The codebase is highly decoupled into sequential, modular scripts to ensure strict computational reproducibility:

* **`00_Data_Simulation/`**: Data construction via a multi-family Gaussian copula sampling framework and adaptive recalibration to build the synthetic cohort.
* **`01_Network_Discovery/`**: Ensemble structural learning using a Genetic Algorithm (GA) optimized via the extended Bayesian Information Criterion (EBIC). **(⏱️ Note: The exhaustive global search in this module requires approximately 4 hours to complete).**
* **`02_Causal_Inference/`**: Implementation of Pearl's *do*-calculus, back-door adjustment, and causal Hazard Ratio (HR) estimations.
* **`03_Visualization/`**: Generation of counterfactual Kaplan-Meier survival curves (G-computation).
* **`04_RealWorld_Validation/`**: Topological validation scripts analyzing independent, real-world multi-omics cohorts (TCGA & MSKCC).
* **`data/`**: Directory for raw summaries, processed individual patient data (IPD), and external validation datasets.

## ⚙️ Environment Setup

To avoid dependency conflicts and ensure exact replication of the study's results, please create a fresh virtual environment (Python 3.10.4 recommended) and install the strictly versioned dependencies:

```bash
# Clone the repository
git clone https://github.com/ZheqingZhu/met14-in-silico-causal.git
cd met14-in-silico-causal

# Install required packages
pip install -r requirements.txt
```

## 🚀 One-Click Reproduction

For peer reviewers and researchers wishing to reproduce the core findings (Table 1, Figure 3, etc.) seamlessly, we provide an automated pipeline script.

Ensure the script has execution privileges, then run it directly from your terminal:

```bash
chmod +x run_pipeline.sh

# Run the standard pipeline
./run_pipeline.sh
```

**Fast-Track Validation (Demo Mode):** The full ensemble network discovery (5 independent PRNG seeds × 100 optimization iterations) is computationally intensive and takes approximately **4 hours**. For a rapid verification of the codebase functionality without waiting for the full global search, use the `--demo` flag. This executes a truncated evolutionary search to validate the pipeline logic in minutes:

```bash
./run_pipeline.sh --demo
```

## 🗄️ Data Schema & Acquisition

### 1. Synthetic Cohort Data Dictionary
When interacting with the generated synthetic data in `data/synthetic/`, please note our standardized schema conventions. All boolean and binary indicator variables strictly utilize the `_available` suffix to ensure high readability and schema consistency. Examples include:
* `event_available`: Binary indicator for survival events (1 = event, 0 = right-censored).
* `at_risk_available`: Indicator for patients remaining in the at-risk pool.
* `censoring_available`: Explicit marker for censored observations.

### 2. External Real-World Validation Data (Figure 5)
To comply with data-sharing agreements and GitHub file size limits, the raw multi-omics validation datasets are **not** included directly in this repository. To completely reproduce the real-world topological validation, please download the public datasets directly from the cBioPortal Datahub:

* **MSKCC LUAD Organotropism (2023):** Download [Archive](https://cbioportal-datahub.s3.amazonaws.com/luad_mskcc_2023_met_organotropism.tar.gz) and extract into `data/raw/luad_mskcc_2023_met_organotropism/`
* **TCGA-LUAD PanCancer Atlas (2018):** Download [Archive](https://cbioportal-datahub.s3.amazonaws.com/luad_tcga_pan_can_atlas_2018.tar.gz) and extract into `data/raw/luad_tcga_pan_can_atlas_2018/`
* **MSKCC NSCLC ctDNA (2022):** Download [Archive](https://cbioportal-datahub.s3.amazonaws.com/nsclc_ctdx_msk_2022.tar.gz) and extract into `data/raw/nsclc_ctdx_msk_2022/`

*(Note: The validation scripts are pre-configured to automatically read the target `.txt` files once placed in these directories.)*

## 🔗 Ecosystem Integration: IPD Reconstruction

The empirical priors and structural constraints utilized in this Copula sampling framework were derived from high-fidelity individual patient data (IPD). Due to the unavailability of shared patient-level data from the original clinical trials, this IPD was computationally reconstructed from published Kaplan-Meier curves using our open-source constraint-consistent inverse optimization framework:

* **Tool:** **[KM-PoPiGo](https://kmpopigo.github.io/)**
* **Transparency:** All raw digital reconstruction project files used in this study have been deposited in this repository. Reviewers can directly import these files into the KM-PoPiGo web interface to interactively verify the curve fits and time-to-event extractions.

## 📝 License & Citation

This project is licensed under the MIT License - see the `LICENSE` file for details. 

*(Citation information will be updated upon publication).*
