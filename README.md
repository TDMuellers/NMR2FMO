# NMR2FMO
Using NMR to predict molecular reactivity, 2026

Authors: Tobias D. Muellers, Predrag V. Petrovic, Paul T. Anastas, Julie B. Zimmerman

## Full citation
To be added

## Contents and overview
<img src="/figures/graphical%20abstract.png" alt="Graphical abstract" width="400">
This repository includes the code required to replicate the analysis conducted in Muellers et al.'s publication, "Using NMR to predict molecular reactivity".

The repository includes:
- A Jupyter notebook with the full machine learning workflow
- A Python script with various functions called by the Jupyter notebook.

The code does not include the specific scripts or commands used to create Gaussian 16 input files nor commands required to run these files. An overview of the methods used are included in the methods section of the manuscript.

The data required to run this code is stored as indicated below.

## Environment 
The environment used to run this code was built using the following commands in series:
1. conda create -n rapids-26.06 -c rapidsai -c conda-forge cudf=26.06 cuml=26.06 python=3.14 'cuda-version>=12.2,<=12.9' 'pytorch=*=*cuda*'
2. conda activate rapids-26.06
3. conda install scikit-learn numpy pandas scipy matplotlib seaborn ipython jupyter jupyterlab tqdm xgboost shap joblib rdkit optuna skorch

## Data location
The data required to run this script is located at: https://dataverse.yale.edu/dataverse/NMR2FMO