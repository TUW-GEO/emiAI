# -*- coding: utf-8 -*-
import os

import numpy as np
from tqdm import tqdm

from forward_utils import computeForwardModel


def main():

    PATH_DATA = "../data/dl/training_data/"
    DEPTH_MODELS = ['1.5m_12layers',
                    '1.5m_6layers',
                    '1.5m_4layers',
                    ]

    FWD_MODEL = "CS"
    # FWD_MODEL = "FSeq"
    # settings are for GF Instruments CMD-MiniExplorer
    COILS = ['VCP0.32f30000h0', 'VCP0.71f30000h0', 'VCP1.18f30000h0',
             'HCP0.32f30000h0', 'HCP0.71f30000h0', 'HCP1.18f30000h0']
    NOISE = 0.0

    for idx in tqdm(range(len(DEPTH_MODELS))):
        depth_model = DEPTH_MODELS[idx]

        print("\nComputing forward response for {}".format(depth_model))
        print("... this will take some time.")
        composite_model = np.load(PATH_DATA + depth_model
                                  + os.sep + "composite_model.npz",
                                  allow_pickle=True)
        # first and last depth are omitted as needed for emagpy
        layer_depths = composite_model["composite_model_depth"]
        # 1D profiles are still electrical resistivity in Ohmm
        ER = composite_model["composite_model"]
        EC = 1 / ER  # EC in S/m

        ECa_vcp, ECa_hcp = computeForwardModel(EC,
                                               layer_depths,
                                               forward_model=FWD_MODEL,
                                               coils=COILS,
                                               noise=NOISE)

        print("Saving the forward response to {}".format(PATH_DATA
                                                         + depth_model))
        np.save(PATH_DATA + depth_model + os.sep + 'EC.npy', EC)
        np.save(PATH_DATA + depth_model + os.sep + 'ECa_hcp_mini.npy', ECa_hcp)
        np.save(PATH_DATA + depth_model + os.sep + 'ECa_vcp_mini.npy', ECa_vcp)


if __name__ == "__main__":
    main()
