========================================================================
Dataset: MSKCC NSCLC ctDNA Cohort (Jee et al., Nature Medicine 2022)
========================================================================

This directory is intentionally left empty to comply with data-sharing 
agreements and repository size limitations. 

To successfully execute the corresponding validation script 
(`04_RealWorld_Validation/realworld_validation2022.py`), you must download 
the raw clinical and genomic data from the cBioPortal Datahub and place 
the extracted text files into this directory.

DOWNLOAD & SETUP INSTRUCTIONS:
------------------------------------------------------------------------
1. Download the study archive directly via this URL:
   https://datahub.assets.cbioportal.org/nsclc_ctdx_msk_2022.tar.gz

2. Extract the downloaded `.tar.gz` archive on your local machine.

3. Locate and copy the following specific files into THIS directory:
   - data_clinical_patient.txt
   - data_clinical_sample.txt
   - data_mutations.txt
   - data_cna.txt

Ensure the filenames remain exactly as listed above, as the topological 
validation pipeline is configured to parse these specific strings.

[!] IMPORTANT: Do not commit the downloaded `.txt` files to version 
control. They are ignored by the root `.gitignore` configuration.
