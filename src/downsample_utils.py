# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm


def computeLayerMids(depth_profile: np.array):
    """Computes the depth layer middle points from a given depth profile.

    Args:
        depth_profile (np.array): Contains the depth model edges.

    Returns:
        layer_mids: Computes layer midpoints.
    """
    layer_mids = pd.DataFrame(depth_profile).rolling(2).mean().values
    return layer_mids[1:]


def downsampleCompositeModel(composite_model: np.array,
                             original_midpoints: np.array,
                             new_midpoints: np.array):
    """Performs downsampling of EC/ER models for a given depth model.

    Args:
        composite_model (np.array): Contains all models which should be
                                    downsampled.
        original_midpoints (np.array): Midpoints of the original depth model.
        new_midpoints (np.array): Midpoints of the new depth model.

    Returns:
        downsampled_models (np.array): Contains the downsampled models.
    """
    downsampled_models = np.empty((len(composite_model),
                                   len(new_midpoints)))
    for idx in tqdm(range(len(downsampled_models))):
        curr_model = composite_model[idx, :]
        downsampled = np.interp(new_midpoints.flatten(),
                                original_midpoints.flatten(),
                                curr_model.flatten())
        downsampled_models[idx, :] = downsampled

    return downsampled_models