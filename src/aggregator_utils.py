# -*- coding: utf-8 -*-
import numpy as np


def loadAndAggregate(path_models: str):
    """_summary_

    Args:
        path_models (str): Filepath to .npz file that contains the generated
                           models for the profile.

    Returns:
        models_out (np.array): Array containing all generated new models for
                               all available extraction points along
                               the profile.
    """

    data = np.load(path_models, allow_pickle=True)
    models_in = data['generated_layer_values']

    # get dimensions of output array
    # number of extraction points along this profile
    nextr_points = len(models_in)
    nlayers, nmodels_per_point = np.shape(models_in[0])

    # construct a new array that will contain all models for this profile
    models_out = np.empty((nextr_points * nmodels_per_point, nlayers))

    idx = 0
    for extr_point in range(0, nextr_points):
        curr_point = models_in[extr_point]
        models_out[idx:idx + nmodels_per_point, :] = curr_point.T
        idx += nmodels_per_point

    return models_out


def joinAllSections(path_list: list):
    """_summary_

    Args:
        path_list (list): Contains the filepaths to the different
                          profiles where new models have been generated.

    Returns:
        composite_model (np.array): Contains the all models for all profiles
                                    and extraction points.
        composite_model_depth (np.array): Contains the depth model edges.
    """

    # load the first section to derive number of layers
    sec1 = loadAndAggregate(path_list[0])
    composite_model_depth = np.load(path_list[0],
                                    allow_pickle=True)['binned_data_depth']
    composite_model = sec1

    # now iterate over the other models and stack the models
    for path_section in path_list[1:]:
        secx = loadAndAggregate(path_section)
        composite_model = np.vstack((composite_model, secx))

    return composite_model, composite_model_depth