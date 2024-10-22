import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from LS_forward_model import *


def prepareEMIdata4DLN(path_emi: str, path_out: str):
    """Reads the EMI raw data and saves them as numpy file in S/m.

    Args:
        path_emi (str): File path to EMI data.
        path_out (str): File path (output name) to save the numpy file.
    """
    df = pd.read_csv(path_emi, sep=',')

    data_out = np.empty((len(df), 3))
    data_out[:, 0] = df['Cond.1 [mS/m]'] / 1000
    data_out[:, 1] = df['Cond.2 [mS/m]'] / 1000
    data_out[:, 2] = df['Cond.3 [mS/m]'] / 1000
    np.save(path_out, data_out)


def forwardResponseForPredictedModel(ECmodel_predicted, depth, **kwargs):
    """
    """
    if np.shape(ECmodel_predicted)[1] > np.shape(ECmodel_predicted)[0]:
        ECmodel_predicted = ECmodel_predicted.T

    output_len = len(ECmodel_predicted)
    coiloffset = kwargs.get('coiloffset', [0.32, 0.71, 1.18])

    ECa_hcp = np.empty((output_len, 3))
    ECa_vcp = np.empty((output_len, 3))

    for idx_off, offset in enumerate(coiloffset):

        for idx in range(0, output_len):

            print('Processing model: {}/{}'.format(idx, output_len))

            hcp, vcp = computeForwardResponse(
                depth, ECmodel_predicted[idx, :], offset, 2 * np.pi * 10000)

            ECa_hcp[idx, idx_off] = hcp
            ECa_vcp[idx, idx_off] = vcp

    return ECa_vcp, ECa_hcp


def prepareInputOutput(input: np.array, output: np.array, nfeatures=1):
    """Prepare the input and output data for DLN. This contains: (1) shuffling
    the arrays, (2) splitting the input and output data sets into training,
    validatio and test data sets, (3) normalization of the inputs and outputs
    and (4) reshaping of the normalized input arrays for the DLN.

    Args:
        input (np.array): Contains the input data.
        output (np.array): Contains the output data.
        nfeatures (int, optional): Number of features. Required for reshaping.
                                   Defaults to 1.

    Returns:
        scalers (list): sklearn.preprocessing.MinMaxScaler functions for
                        input and output normalization.
        inputs (list): Contains the shuffled and split input training,
                       validation and test data sets prior to normalization and
                       reshaping.
        outputs (list): Contains the shuffled and split output training,
                        validation and test data sets prior to normalization.
        inputs_norm (list): Contains the shuffled and split input training,
                            validation and test data sets after normalization
                            and reshaping.
        outputs_norm (list): Contains the shuffled and split input training,
                             validation and test data sets after normalization
                             and reshaping.
    """
    # shuffle input and output data
    input, output = shuffleArrays(input, output)
    # Split data into training, validation and test data sets
    input_train, input_val, input_test = splitArraysByPercentage(input,
                                                                 0.7,
                                                                 0.15)
    output_train, output_val, output_test = splitArraysByPercentage(output,
                                                                    0.7,
                                                                    0.15)
    # normalize the different subsets and keep scaler function at hand
    print(output_train)
    print(output_val)
    print(output_test)

    scaler_in, inputs_norm = normalizeArrays(
        input,
        [input_train, input_val, input_test])
    scaler_out, outputs_norm = normalizeArrays(
        output,
        [output_train, output_val, output_test])

    scalers = [scaler_in, scaler_out]
    inputs = [input_train, input_val, input_test]
    outputs = [output_train, output_val, output_test]

    # finally prepare the input arrays for the DLN
    inputs_norm = [reshapeInputArrays(subset, nfeatures)
                   for subset in inputs_norm]

    return scalers, inputs, outputs, inputs_norm, outputs_norm


def reshapeInputArrays(X: np.array, nfeatures: int):
    """Reshapes the input data set for the DLN with the required number of
    features (here 1).

    Args:
        X (np.array): Input array for reshaping.
        nfeatures (int): Number of features.

    Returns:
        Xrs (np.array): Reshaped input array.
    """
    Xrs = X.reshape((X.shape[0], X.shape[1], nfeatures))
    return Xrs


def normalizeArrays(input: np.array, subsets: list):
    """Normalize the subset arrays with input as reference to the feature
    range (0, 1).

    Args:
        input (np.array): Array which is the reference for normalization.
        subsets (list): Contains the arrays which should be normalized.

    Returns:
        scaler (sklearn.preprocessing.MinMaxScaler): Scaling function.
        normalized_subset (list): Contains the arrays after normalization.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(input)
    normalized_subsets = [scaler.transform(subset) for subset in subsets]
    return scaler, normalized_subsets


def splitArraysByPercentage(input: np.array,
                            perc_train: float,
                            perc_validate: float):
    """Split the input array into training, validation and test data sets. The
    percentage of the test data set is derived from the validation and training
    percentages (100-perc_train-perc_validate).

    Args:
        input (np.array): Array that should be split.
        perc_train (float): Percentage for the training data set.
        perc_validate (float): Percentage for the validation data set.

    Returns:
        train (np.array): Array containing perc_train % of input array.
        validate (np.array): Array containing perc_validate % of input array.
        test (np.array): Array containing (100-perc_train-perc_validate) % of
                         input array.
    """
    train, rest = np.split(input, [int(perc_train * len(input))])
    test, validate = np.split(rest, [int(perc_validate * len(input))])
    return train, validate, test


def findModelNumber(path_model):

    num_models_run = len(glob.glob(path_model + '*'))
    if num_models_run == 0:
        model_index = '1'
    else:
        model_index = str(num_models_run + 1)
    return model_index


def shuffleArrays(X: np.array, Y: np.array):
    """Permute two arrays in the same random way.

    Args:
        X (np.array): Data array 1.
        Y (np.array): Data array 2.

    Returns:
        Xp, Yp (np.array, np.array): Permutet arrays.
    """

    rng = np.random.default_rng(seed=42)
    permutation = rng.permutation(np.shape(X)[0])
    return X[permutation, :], Y[permutation, :]


# function to predict EC data using CNN model
def performModelPrediction(ECa_input: np.array,
                           model,
                           scaler_in,
                           scaler_out):
    """For a given model perform the prediction for the passed ECa values. 
    For a sucessful prediction the input needs to first normalized and
    reshaped. After model prediction the obtained EC model values need to be 
    retransformed with the output scaler (scaler_out).

    Args:
        ECa_input (np.array): Contains the ECa values.
        model (keras.Model()): DLN to perform the prediction.
        scaler_in (sklearn.preprocessing.MinMaxScaler): Input scaler.
        scaler_out (sklearn.preprocessing.MinMaxScaler): Output scaler.

    Returns:
        EC (np.array): Predicted EC depth model.
    """
    # input ECa values need to be normalized, reshaped, then predicted
    # and then retransformed
    ECa_norm = scaler_in.transform(ECa_input.T)
    ECa_norm_reshaped = reshapeInputArrays(ECa_norm, 1)
    EC_norm = model.predict(ECa_norm_reshaped)
    EC = scaler_out.inverse_transform(EC_norm)
    return EC


def joinPredictionAndCoordinates(path_raw: str, path_out: str,
                                 EC: np.array):
    """Merges the prediction results of the model, i.e. the EC values of the 
    1D depth model with the coordinates contained in the raw input file.

    Args:
        path_raw (str): Path to raw data containing the coordinates.
        path_out (str): File path (output name) of the joined file.
        EC (np.array): Contains the n 1D EC models and represents the
                       prediction results of the model.

    Returns:
        df_out (pd.DataFrame): Contains the Longitude, Latitude and Altitude of 
                               the prediction results as well as the x EC
                               values comprising the 1D model.
    """

    df = pd.read_csv(path_raw, sep=',')
    df_coords = df[['Latitude', 'Longitude', 'Altitude']]
    depth_labels = ['D' + str(idx) for idx in range(np.min(np.shape(EC)))]
    df_pred = pd.DataFrame(EC, columns=depth_labels)
    df_out = df_coords.copy()
    df_out[depth_labels] = df_pred.loc[:, depth_labels] * 1000

    df_out.to_csv(path_out,
                  sep=',',
                  index=False)
    return df_out
