![emiAI logo](logo/emiai_medium.png)

# emiAI

emiAI is a framework for the "inversion" of electromagnetic 
induction imaging (EMI) data based on a deep learning network written in Python using
Keras and Tensorflow. It covers the generation training data, the training and evaluation of the deep learning (DL) network and the 
prediction of 1D vertical electrical conductivity (EC) models from measured 
vertical coplanar (VCP) apparent electrical conductivity (ECa) data. 

The framework basically takes care of the following steps:
1. Generation of training models for the deep learning network from electrical
resistivity tomography (ERT) data. The training models consist of 3 measured VCP ECa values and 
EC depth models with 4, 6 and 12 depth layers.
2. Training and evaluation of the DL network
3. Prediction of the 1D vertical EC models. 

## Installation

Install the required packages and dependencies in new conda environment:

> ```bash
> conda env create -f environment.yml
> ```

This work would not have been possible withou many other open-source libraries, so please have also a look at the following repositories and consider citing the corresponding articles:

https://gitlab.com/hkex/resipy

https://gitlab.com/hkex/emagpy

https://github.com/keras-team/keras

## Usage

The workflow consists of running the numbered Python scripts in sequence:

> ```bash
> python 01_generate_1Dmodels.py
> python 02_compute_forward_response.py
> python 03_DLN_train.py
> python 04_DLN_predict.py
> ```

The script `01_generate_1Dmodels.py` does parse a .json file (`config.json`) containing settings and some keyword parameters can be set upon calling the script. For all other scripts, parameters have to be changed in the scripts directly. 

The keyword parameters in `01_generate_1Dmodels.py` are:

- `--plot`: Plot the figures during model generation. This includes the newly 
random generated models, the binning results as well as indication of the sampling
locations. 
- `--aggregate`: Upon model generation aggregate all models for each profile into
a composite model. This composite model can be used for downsampling and is the 
input for the forward computation and training of the DL network.
- `--aggregate_only`: Only aggregate the newly generated models without new model 
generation. This option uses the in the `config.json` specified profiles. 
- `--downsample`: The DL network requires the training data for 3 different depth 
models with 12, 6 and 4 layers. This option downsamples the composite model for the in 
`config.json` specified models. 

The different options in `config.json` are: 

- `PATH_DATA`: Specifies the path to the folder that contains the individual ERT inversion results.
- `PATH_DATA`: Specifies the path to the folder where the modelled training data should be stored.
- `FID_LIST`: Specifies the names of the individual ERT profiles. Can be used with `--aggregate_only` to only aggregate models for specified profiles.
- `DEPTH`: Maximum depth for which ERT data should be extracted. Also specifies the maximum depth of the 12 (or other) layer depth model. 
- `NLAYERS`: Number of depth layers for the largest depth model (before downsampling). 
- `NMODELS`: Number of newly generated models per sampled location in each profile. 
- `NEXTRACTION_RANGE`: How many locations in each profile should be sampled. Number is randomly drawn from range. 
- `SIDE_SKIP`: Number of electrode positions to each side of the profile to be skipped because it is associated to low sensitive regions in the ERT profile. 
- `EXTRACTION_WIDTH_SCALER`: Multiplies the used electrode spacing by this factor to increase the width of the sampling location. 
- `LAYER_EDGES`: Defines the upper and lower limits of each depth layer in the 12 (or other) layer depth model. 
- `LAYER_EDGES_6`: Defines the upper and lower limits of each depth layer in the 6 layer depth model.
- `LAYER_EDGES_4`: Defines the upper and lower limits of each depth layer in the 4 layer depth model.
- `TOTAL_OFFSET`: Specifies an total offset all layer values of the newly generated models are shifted.
- `N0_THICKNESS`: Specifies the minimum thickness of the shallowest layer of the 12 (or other) depth layer model to ensure an adequate number of samples in it. 
- `N0_OFFSET`: Total offset for the shallowest layer that shifts the values of the newly generated model values. 
- `N0_SCALER`: Scales the standard deviation for the shallowest layer.
- `NEND_SCALER`: Scales the standard deviation for the deepest layer.
- `RESTRICT_NEND`: Specifies if a model change direction for last bin/layer is allowed and uses the direction from the one above.
- `DISTRIBUTION`: From which distribution should the new layer values be drawn. Normal or uniform. 
- `SMOOTHER`: Specifies the smoothing operator which is used for smoothing the binned as well as the newly generated models. Can be either convolve or savgol.  
- `BOUNDS`: Specifies the minimum and maximum allowed values for new models. If the generated values exceed these limits they are downscaled to fit into the bounds. 
- `EXAG_FACTORS`: Scaling factors for the standard deviations obtained in binning. Larger values permit more variation in the newly generated models.   
- `REVERSE`: Generate models in forward and reverse direction. 
- `ER_PLOT_RANGE`: Specifies electrical resitivity colorbar and plot range for the generated plots. 


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

