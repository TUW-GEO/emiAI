# -*- coding: utf-8 -*-
import numpy as np
from emagpy import Problem


def computeForwardModel(EC, layer_depths,
                        forward_model='CS',
                        coils=None,
                        noise=0.00):
    """For the given 1D EC models in np.array EC compute the forward response
    with EMagPy and the given settings.

    Args:
        EC (np.array): Contains the 1D EC depth models (n x p).
        layer_depths (np.array): Contains the layer depths of the
                               1D model(p+1 x 1).
        forward_model (str, optional): Forward model used in EMagPy.
                                       Defaults to 'CS'.
        coils (list, optional): Contains the coil spacings for VCP and HCP and
                                the operating frequency of the sensor.
                                Defaults to None and uses CMD-MiniExplorer
                                settings.
        noise (float, optional): Noise added during forward modeling.
                                 Defaults to 0.00.

    Returns:
        ECa_vcp, ECa_hcp (np.array, np.array): Forward response (ECa values)
                                               for the provided EC models.
    """

    if coils is None:
        # then default to MiniExplorer
        coils = ['VCP0.32f30000h0', 'VCP0.71f30000h0', 'VCP1.18f30000h0',
                 'HCP0.32f30000h0', 'HCP0.71f30000h0', 'HCP1.18f30000h0']

    output_len = len(EC)
    ECa_hcp = np.empty((output_len, 3))
    ECa_vcp = np.empty((output_len, 3))

    # needs to be a (1, n, 1) array for EMagPy
    depths = np.ones((1, output_len, 1))*layer_depths[1:-1]
    models = np.expand_dims(EC*1000, axis=0)  # EC in mS/m

    k = Problem()  # create an instance of the Problem class
    df = k.forward(forwardModel=forward_model,
                   coils=coils,
                   noise=noise,
                   models=models,  # input in mS/m
                   depths=depths)

    ECa_vcp = df[0][coils[0:3]].values / 1000  # back to S/m
    ECa_hcp = df[0][coils[3:]].values / 1000

    return ECa_vcp, ECa_hcp
