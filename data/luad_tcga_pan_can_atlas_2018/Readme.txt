========================================================================
Dataset: TCGA-LUAD PanCancer Atlas Cohort (Cell 2018)
========================================================================

This directory is intentionally left empty to comply with data-sharing 
agreements and repository size limitations. 

To successfully execute the corresponding real-world validation scripts 
within the `04_RealWorld_Validation/` module, you must download the raw 
clinical and transcriptomic data from the cBioPortal Datahub and place 
the extracted text files into this directory.

DOWNLOAD & SETUP INSTRUCTIONS:
------------------------------------------------------------------------
1. Download the study archive directly via this URL:
   https://datahub.assets.cbioportal.org/luad_tcga_pan_can_atlas_2018.tar.gz

2. Extract the downloaded `.tar.gz` archive on your local machine.

3. Locate and copy EXACTLY the following two files into THIS directory:
   - data_clinical_sample.txt
   - data_mrna_seq_v2_rsem.txt

Ensure the filenames remain exactly as listed above, as the topological 
validation pipeline is configured to parse these specific strings.

[!] IMPORTANT: Do not commit the downloaded `.txt` files to version 
control. They are ignored by the root `.gitignore` configuration.
