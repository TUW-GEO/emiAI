![emiAI logo](logo/emiai_medium.png)

# emiAI

emiAI is a framework for the "inversion" of electromagnetic 
induction imaging (EMI) data based on a deep learning network written in Python using
Keras and Tensorflow. It covers the generation training data, the training and evaluation of the deep learning (DL) network and the 
prediction of 1D vertical electrical conductivity (EC) models from measured 
vertical coplanar (VCP) apparent electrical conductivity (ECa) data. 

The framework basically takes care of the following steps:
1. Generation of training models for the deep learning network from electrical
resistivity data. The training models consist of 3 measured VCP ECa values and 
EC depth models with 4, 6 and 12 depth layers.
2. Training and evaluation of the DL network
3. Prediction of the 1D vertical EC models. 

## Installation

can be installed by 

> ```bash
> conda env create -f environment.yml
> ```

However important contributions are 

https://gitlab.com/hkex/resipy
https://gitlab.com/hkex/emagpy
https://github.com/keras-team/keras

## Usage

state the workflow and what the scripts do, where can changes be done

explain the settings in generator script in detail 

## Citation ##

[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.4610836.svg)](https://doi.org/10.5281/zenodo.4610836)

If you use the software in a publication then please cite it using the Zenodo
DOI. Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.4610836 to
get the DOI of that version. You should normally always use the DOI for the
specific version of your record in citations. This is to ensure that other
researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at
http://help.zenodo.org/#versioning

