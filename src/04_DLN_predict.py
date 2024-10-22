# -*- coding: utf-8 -*-
import os

import joblib
import numpy as np
from tensorflow import keras

from DLN_utils import (joinPredictionAndCoordinates, performModelPrediction,
                       prepareEMIdata4DLN)
from forward_utils import computeForwardModel
from plotting_utils import getDepthLabels, plotPredictedECMaps

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():

    CLIMITS2 = [1, 50]
    FWD_MODEL = 'CS'

    PATH_FIELD_DATA = '../data/emi/all_data_wgs_2016_nonnegative.csv'
    PATH_FIELD_DATA_C = '../data/emi/ECa_vcp_2016'
    PREDICTION_OUT = "dln_prediction_2016.txt"
    prepareEMIdata4DLN(PATH_FIELD_DATA,
                       PATH_FIELD_DATA_C)
    eca_vcp = np.load(PATH_FIELD_DATA_C + ".npy")

    PATH_MODELS = ["../data/dl/models/1.5m_4layers",
                   "../data/dl/models/1.5m_6layers",
                   "../data/dl/models/1.5m_12layers"
                   ]
    MODEL_IDF = ['4x1', '6x1', '12x1']
    FWD = []

    for midx, path_model in enumerate(PATH_MODELS):
        print("Processing model: {}".format(path_model))
        # load scalers
        scaler_in = joblib.load(path_model + os.sep
                                + "scaler_in_{}.pkl".format(MODEL_IDF[midx]))
        scaler_out = joblib.load(path_model + os.sep
                                 + "scaler_out_{}.pkl".format(MODEL_IDF[midx]))
        model = keras.models.load_model(
            path_model
            + os.sep
            + 'dln_model_{}.keras'.format(MODEL_IDF[midx]),
            compile=False)
        depth = np.loadtxt(
            path_model
            + os.sep
            + 'layer_depths_{}.txt'.format(MODEL_IDF[midx]))
        if midx == 0:
            # subnet 1
            prediction = performModelPrediction(eca_vcp.T,
                                                model,
                                                scaler_in,
                                                scaler_out)
        elif midx == 1:
            # subnet 2: 3x vcp + 3x hcp
            prediction = performModelPrediction(FWD[midx-1].T,
                                                model,
                                                scaler_in,
                                                scaler_out)
        elif midx == 2:
            # subnet 3: 3x vcp + 3x hcp
            prediction = performModelPrediction(FWD[midx-1].T,
                                                model,
                                                scaler_in,
                                                scaler_out)
        np.savetxt(path_model + os.sep + PREDICTION_OUT,
                   prediction.T,
                   delimiter='\t')
        df = joinPredictionAndCoordinates(
            PATH_FIELD_DATA,
            path_model + os.sep + PREDICTION_OUT.replace('.txt',
                                                         '_WGS.csv'),
            prediction)
        depth_label = getDepthLabels(depth)
        plotPredictedECMaps(
            df, depth_label, CLIMITS2,
            path_model + os.sep + PREDICTION_OUT.replace('.txt', ''))
        if MODEL_IDF[midx] in ['4x1', '6x1']:
            forward_vcp, forward_hcp = computeForwardModel(
                prediction, depth,
                forward_model=FWD_MODEL)
            FWD.append(np.hstack([forward_vcp, forward_hcp]))


if __name__ == "__main__":
    main()
