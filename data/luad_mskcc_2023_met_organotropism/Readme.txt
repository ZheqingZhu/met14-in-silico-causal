========================================================================
Dataset: MSKCC LUAD Organotropism Cohort (Cancer Cell 2023)
========================================================================

This directory is intentionally left empty to comply with data-sharing 
agreements and repository size limitations. 

To successfully execute the real-world macroscopic penalty validation script 
(e.g., generating Fig 5), you must download the raw clinical data from the 
cBioPortal Datahub and place the extracted text file into this directory.

DOWNLOAD & SETUP INSTRUCTIONS:
------------------------------------------------------------------------
1. Download the study archive directly via this URL:
   https://datahub.assets.cbioportal.org/luad_mskcc_2023_met_organotropism.tar.gz

2. Extract the downloaded `.tar.gz` archive on your local machine.

3. Locate and copy EXACTLY the following file into THIS directory:
   - data_clinical_patient.txt

Ensure the filename remains exactly as listed above, as the topological 
validation pipeline is configured to parse this specific file to extract 
OS_MONTHS, OS_STATUS, CNS_STATUS, and LIVER_STATUS.

[!] IMPORTANT: Do not commit the downloaded `.txt` file to version 
control. It is ignored by the root `.gitignore` configuration.
