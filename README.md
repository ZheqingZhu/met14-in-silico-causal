# Inferring the Causal Architecture of METex14 NSCLC via a Synthetic Cohort

[![Python 3.10.4](https://img.shields.io/badge/python-3.10.4-blue.svg)](https://www.python.org/downloads/release/python-3104/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This repository contains the official codebase and analytical pipeline for our study investigating the causal architecture and therapeutic mediators of *MET*ex14 non-small cell lung cancer (NSCLC). 

The evaluation of rare mutational subsets is typically constrained by data fragmentation and multidimensional confounding. To address this challenge, we provide a fully reproducible computational framework. Specifically, we apply a multi-family Gaussian copula sampling framework to construct a 3,000-patient synthetic cohort. We then implement an ensemble score-based causal discovery algorithm (regularized by EBIC) and Pearl's *do*-calculus to isolate absolute survival effects from baseline confounding.

## 🏗️ Repository Architecture

The analytical pipeline is organized into sequential modules to facilitate computational reproducibility:

* **`00_Data_Simulation/`**: Data construction and adaptive recalibration via a multi-family Gaussian copula sampling framework to build the synthetic cohort.
* **`01_Network_Discovery/`**: Ensemble structural learning using a Genetic Algorithm (GA) optimized via the extended Bayesian information criterion (EBIC).
* **`02_Causal_Inference/`**: Implementation of Pearl's *do*-calculus, back-door adjustment, and causal hazard ratio (HR) estimation.
* **`03_Visualization/`**: Construction of counterfactual Kaplan-Meier survival curves via parametric G-computation.
* **`04_RealWorld_Validation/`**: Topological validation analyzing independent, real-world multi-omics cohorts (TCGA & MSKCC).
* **`data/`**: Directory containing raw summary statistics, reconstructed individual patient data (IPD), and external validation datasets.

## ⚙️ Environment Setup

We recommend configuring a clean virtual environment (Python 3.10.4) and installing the pinned dependencies to match our exact computational environment:

```bash
# Clone the repository
git clone [https://github.com/ZheqingZhu/met14-in-silico-causal.git](https://github.com/ZheqingZhu/met14-in-silico-causal.git)
cd met14-in-silico-causal

# Install required packages
pip install -r requirements.txt
```

## 🚀 One-Click Reproduction

For peer reviewers and researchers wishing to reproduce the core findings (Table 1, Figure 3, etc.) seamlessly, we provide a cross-platform automated pipeline script.

You can execute the entire pipeline directly from your terminal:

```bash
# Run the standard pipeline
python main.py
```

**(⏱️ Note on Execution Time):** The standard pipeline includes the full ensemble network discovery (5 independent PRNG seeds × 100 optimization iterations) within the `01_Network_Discovery` module. This specific step is computationally intensive and requires approximately **4 hours** to complete on standard hardware.

**Fast-Track Validation (Demo Mode):** For a rapid verification of the codebase functionality without waiting for the exhaustive optimization process, use the `--demo` flag. This executes a truncated evolutionary search to validate the pipeline logic in just a few minutes:

```bash
python main.py --demo
```

## 🗄️ Data Schema & Acquisition

### 1. Synthetic Cohort Data Dictionary
When interacting with the generated synthetic data in `data/synthetic/`, please note our standardized schema conventions. All boolean and binary indicator variables strictly utilize the `_available` suffix to ensure high readability and schema consistency. Examples include:
* `event_available`: Binary indicator for survival events (1 = event, 0 = right-censored).
* `at_risk_available`: Indicator for patients remaining in the at-risk pool.
* `censoring_available`: Explicit marker for censored observations.

### 2. External Real-World Validation Data (Figure 5)
To comply with data-sharing agreements and GitHub file size limits, the raw multi-omics validation datasets are **not** included directly in this repository. To completely reproduce the real-world topological validation, please download the public datasets directly from the cBioPortal Datahub:

* **MSKCC LUAD Organotropism (2023):** Download [Archive](https://datahub.assets.cbioportal.org/luad_mskcc_2023_met_organotropism.tar.gz) and extract into `data/raw/luad_mskcc_2023_met_organotropism/`
* **TCGA-LUAD PanCancer Atlas (2018):** Download [Archive](https://datahub.assets.cbioportal.org/luad_tcga_pan_can_atlas_2018.tar.gz) and extract into `data/raw/luad_tcga_pan_can_atlas_2018/`
* **MSKCC NSCLC ctDNA (2022):** Download [Archive](https://datahub.assets.cbioportal.org/nsclc_ctdx_msk_2022.tar.gz) and extract into `data/raw/nsclc_ctdx_msk_2022/`

*(Note: The validation scripts are pre-configured to automatically read the target `.txt` files once placed in these directories.)*

## 📝 License & Citation

This project is licensed under the MIT License - see the `LICENSE` file for details. 

*(Citation information will be updated upon publication).*
