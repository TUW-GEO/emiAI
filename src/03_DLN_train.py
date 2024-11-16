# -*- coding: utf-8 -*-

import os
import pickle

import joblib
import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DLN_utils import performModelPrediction, prepareInputOutput
from forward_utils import computeForwardModel
from plotting_utils import (plotAll1DModels,
                            plotRegressionsTrainingAndPredictedEC,
                            plotTrainingEvolution)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def trainSubnet4x1(path_input: str, path_out: str,
                   epochs=512,
                   batchsize=512):
    """Sets up the subnet 1 and trains it.
    The input are 3 ECa VCP values and the output are 4 layer EC depth models.

    Args:
        path_input (str): Path to input and output data = training data set.
        path_out (str): Path to output folder where model and evaluation etc.
                        should be saved.
        epochs (int, optional): Number of epochs for training. Defaults to 512.
        batchsize (int, optional): How large the training batch should be.
                                   Defaults to 512.

    Returns:
        [model, ECa, scalers] (list): Contains the following:
         model (keras.Model()): Trained DL model.
         ECa (np.array): Contains the ECa values used to train the model.
         scalers (sklearn.preprocessing.MinMaxScaler): Scalers for input and
                                                       output.
        [tr_history, evaluation] (list): Contains the following:
         tr_history (pd.DataFrame): Contains the loss and other training
                                    metrics and their change during training
                                    for the training and validation set.
         evaluation (list): Contains the loss, and other specified metrics for
                            the test data set.
    """

    # generate the output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # load input and output and prepare arrays
    ECa = np.load(path_input + os.sep + 'ECa_vcp_mini.npy',
                  allow_pickle=True)
    ECr = np.load(path_input + os.sep + 'EC.npy',
                  allow_pickle=True)
    # load depth layers and save later to have reference to used depth
    depth_layers = np.load(path_input + 'composite_model.npz',
                           allow_pickle=True)['composite_model_depth']

    input = ECa.copy()
    output = ECr.copy()

    scalers, inputs, outputs, inputs_norm, outputs_norm = prepareInputOutput(
        input, output)

    # unpack the lists
    scaler_in, scaler_out = scalers
    input_train, input_val, input_test = inputs
    output_train, output_val, output_test = outputs
    input_train_norm, input_val_norm, input_test_norm = inputs_norm
    output_train_norm, output_val_norm, output_test_norm = outputs_norm

    # SETUP THE DLN AND SETTINGS
    # training setting
    opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    # opt = keras.optimizers.Adam(learning_rate=0.0001)
    str_metrics = ['mean_squared_error', 'mean_absolute_error', 'accuracy']
    str_labels = ['MSE', 'MAE', 'Accuracy']
    loss = 'mae'

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=5,
                                      restore_best_weights=True)]

    # seed(0)
    # tensorflow.random.set_seed(0)

    model = keras.Sequential()
    model.add(keras.Input(shape=(3,)))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(4, activation=None))
    model.add(keras.layers.Flatten())
    print(model.summary())

    # compile, fit and evaluate model
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=str_metrics)

    history = model.fit(input_train_norm,
                        output_train_norm,
                        epochs=epochs,
                        batch_size=batchsize,
                        validation_data=(input_val_norm, output_val_norm),
                        verbose=1,
                        callbacks=callbacks)
    # training history; prepare pd.DataFrame for plotting later
    tr_history = pd.DataFrame(history.history)
    tr_history['epoch'] = history.epoch

    evaluation = model.evaluate(input_test_norm,
                                output_test_norm,
                                batch_size=batchsize,
                                callbacks=callbacks,
                                return_dict=False)
    print(evaluation)

    # Make predictions for later evaluation
    train_predictions_norm, val_predictions_norm, test_predictions_norm = [
        model.predict(subset) for subset in [input_train_norm,
                                             input_val_norm,
                                             input_test_norm]]
    train_predictions, val_predictions, test_predictions = [
        scaler_out.inverse_transform(subset)
        for subset in [train_predictions_norm,
                       val_predictions_norm,
                       test_predictions_norm]]

    # construct an output dict
    dln_data = {'input_train': input_train,
                'input_test': input_test,
                'input_val': input_val,
                'output_train': output_train,
                'output_test': output_test,
                'output_val': output_val,
                'input_train_norm': input_train_norm,
                'input_test_norm': input_test_norm,
                'input_val_norm': input_val_norm,
                'output_train_norm': output_train_norm,
                'output_test_norm': output_test_norm,
                'output_val_norm': output_val_norm,
                'test_predictions': test_predictions,
                'val_predictions': val_predictions,
                'train_predictions': train_predictions,
                'str_metrics': str_metrics,
                'str_labels': str_labels}

    # save the data, the trained model, scalers, evaluation metrics and history
    joblib.dump(scaler_in, path_out + os.sep + 'scaler_in_4x1.pkl')
    joblib.dump(scaler_out, path_out + os.sep + 'scaler_out_4x1.pkl')
    np.savez_compressed(path_out + os.sep + 'dln_data_4x1', **dln_data)
    model.save(path_out + os.sep + 'dln_model_4x1.h5')
    model.save(path_out + os.sep + 'dln_model_4x1.keras')
    with open(path_out + os.sep + 'tr_history_4x1.pickle', 'wb') as f:
        pickle.dump(tr_history, f)
    # save the information about for later reference
    np.savetxt(path_out + os.sep + 'layer_depths_4x1.txt',
               depth_layers,
               comments='')
    np.savetxt(path_out + os.sep + 'evaluation_metrics.txt',
               evaluation[1:],
               delimiter=',',
               header=(','.join(str_metrics)),
               comments='')

    return [model, ECa, scalers], [tr_history, evaluation], depth_layers


def trainSubnet6x1(path_input: str,
                   path_out: str,
                   input_ECa_values: np.array,
                   epochs=512,
                   batchsize=512):
    """Sets up the subnet 2 and trains it.
    The input are 3 ECa VCP and 3 HCP values and the output are 6 layer
    EC depth models. The 6 ECa values are based on forward modelling the
    predicted 4 layer EC depth models from the previous subnet (1).

    Args:
        path_input (str): Path to input and output data = training data set.
        path_out (str): Path to output folder where model and evaluation etc.
                        should be saved.
        input_ECa_values (np.array): Contain the 6 ECa values (3 VCP, 3 HCP).
        epochs (int, optional): Number of epochs for training. Defaults to 512.
        batchsize (int, optional): How large the training batch should be.
                                   Defaults to 512.

    Returns:
        [model, ECa, scalers] (list): Contains the following:
         model (keras.Model()): Trained DL model.
         ECa (np.array): Contains the ECa values used to train the model.
         scalers (sklearn.preprocessing.MinMaxScaler): Scalers for input and
                                                       output.
        [tr_history, evaluation] (list): Contains the following:
         tr_history (pd.DataFrame): Contains the loss and other training
                                    metrics and their change during training
                                    for the training and validation set.
         evaluation (list): Contains the loss, and other specified metrics for
                            the test data set.
    """
    # generate the output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # load input and output and prepare arrays
    ECa = input_ECa_values
    ECr = np.load(path_input + os.sep + 'EC.npy',
                  allow_pickle=True)
    # load depth layers and save later to have reference to used depth
    depth_layers = np.load(path_input + 'composite_model.npz',
                           allow_pickle=True)['composite_model_depth']

    input = ECa.copy()
    output = ECr.copy()

    scalers, inputs, outputs, inputs_norm, outputs_norm = prepareInputOutput(
        input, output)

    # unpack the lists
    scaler_in, scaler_out = scalers
    input_train, input_val, input_test = inputs
    output_train, output_val, output_test = outputs
    input_train_norm, input_val_norm, input_test_norm = inputs_norm
    output_train_norm, output_val_norm, output_test_norm = outputs_norm

    # SETUP THE DLN AND SETTINGS
    # training setting
    opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    # opt = keras.optimizers.Adam(learning_rate=0.0001)
    str_metrics = ['mean_squared_error', 'mean_absolute_error', 'accuracy']
    str_labels = ['MSE', 'MAE', 'Accuracy']
    loss = 'mae'

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=5,
                                      restore_best_weights=True)]

    # seed(0)
    # tensorflow.random.set_seed(0)

    model = keras.Sequential()
    model.add(keras.Input(shape=(6,)))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation=None))
    model.add(keras.layers.Flatten())
    print(model.summary())

    # compile, fit and evaluate model
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=str_metrics)

    history = model.fit(input_train_norm,
                        output_train_norm,
                        epochs=epochs,
                        batch_size=batchsize,
                        validation_data=(input_val_norm, output_val_norm),
                        verbose=1,
                        callbacks=callbacks)
    # training history; prepare pd.DataFrame for plotting later
    tr_history = pd.DataFrame(history.history)
    tr_history['epoch'] = history.epoch

    evaluation = model.evaluate(input_test_norm,
                                output_test_norm,
                                batch_size=batchsize,
                                callbacks=callbacks,
                                return_dict=False)
    print(evaluation)

    # Make predictions for later evaluation
    train_predictions_norm, val_predictions_norm, test_predictions_norm = [
        model.predict(subset) for subset in [input_train_norm,
                                             input_val_norm,
                                             input_test_norm]]
    train_predictions, val_predictions, test_predictions = [
        scaler_out.inverse_transform(subset)
        for subset in [train_predictions_norm,
                       val_predictions_norm,
                       test_predictions_norm]]

    # construct an output dict
    dln_data = {'input_train': input_train,
                'input_test': input_test,
                'input_val': input_val,
                'output_train': output_train,
                'output_test': output_test,
                'output_val': output_val,
                'input_train_norm': input_train_norm,
                'input_test_norm': input_test_norm,
                'input_val_norm': input_val_norm,
                'output_train_norm': output_train_norm,
                'output_test_norm': output_test_norm,
                'output_val_norm': output_val_norm,
                'test_predictions': test_predictions,
                'val_predictions': val_predictions,
                'train_predictions': train_predictions,
                'str_metrics': str_metrics,
                'str_labels': str_labels}

    # save the data, the trained model, scalers, evaluation metrics and history
    joblib.dump(scaler_in, path_out + os.sep + 'scaler_in_6x1.pkl')
    joblib.dump(scaler_out, path_out + os.sep + 'scaler_out_6x1.pkl')
    np.savez_compressed(path_out + os.sep + 'dln_data_6x1', **dln_data)
    model.save(path_out + os.sep + 'dln_model_6x1.h5')
    model.save(path_out + os.sep + 'dln_model_6x1.keras')
    with open(path_out + os.sep + 'tr_history_6x1.pickle', 'wb') as f:
        pickle.dump(tr_history, f)
    # save the information about for later reference
    np.savetxt(path_out + os.sep + 'layer_depths_6x1.txt',
               depth_layers,
               comments='')
    np.savetxt(path_out + os.sep + 'evaluation_metrics.txt',
               evaluation[1:],
               delimiter=',',
               header=(','.join(str_metrics)),
               comments='')

    return [model, ECa, scalers], [tr_history, evaluation], depth_layers


def trainSubnet12x1(path_input: str,
                    path_out: str,
                    input_ECa_values: np.array,
                    epochs=512,
                    batchsize=512):
    """Sets up the subnet 3 and trains it.
    The input are 3 ECa VCP and 3 HCP values and the output are 12 layer
    EC depth models. The 6 ECa values are based on forward modelling the
    predicted 6 layer EC depth models from the previous subnet (2).

    Args:
        path_input (str): Path to input and output data = training data set.
        path_out (str): Path to output folder where model and evaluation etc.
                        should be saved.
        input_ECa_values (np.array): Contain the 6 ECa values (3 VCP, 3 HCP).
        epochs (int, optional): Number of epochs for training. Defaults to 512.
        batchsize (int, optional): How large the training batch should be.
                                   Defaults to 512.

    Returns:
        [model, ECa, scalers] (list): Contains the following:
         model (keras.Model()): Trained DL model.
         ECa (np.array): Contains the ECa values used to train the model.
         scalers (sklearn.preprocessing.MinMaxScaler): Scalers for input and
                                                       output.
        [tr_history, evaluation] (list): Contains the following:
         tr_history (pd.DataFrame): Contains the loss and other training
                                    metrics and their change during training
                                    for the training and validation set.
         evaluation (list): Contains the loss, and other specified metrics for
                            the test data set.
    """

    # generate the output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # load input and output and prepare arrays
    ECa = input_ECa_values
    ECr = np.load(path_input + os.sep + 'EC.npy',
                  allow_pickle=True)
    # load depth layers and save later to have reference to used depth
    depth_layers = np.load(path_input + 'composite_model.npz',
                           allow_pickle=True)['composite_model_depth']

    input = ECa.copy()
    output = ECr.copy()

    scalers, inputs, outputs, inputs_norm, outputs_norm = prepareInputOutput(
        input, output)

    # unpack the lists
    scaler_in, scaler_out = scalers
    input_train, input_val, input_test = inputs
    output_train, output_val, output_test = outputs
    input_train_norm, input_val_norm, input_test_norm = inputs_norm
    output_train_norm, output_val_norm, output_test_norm = outputs_norm

    # SETUP THE DLN AND SETTINGS
    # training setting
    opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    # opt = keras.optimizers.Adam(learning_rate=0.0001)
    str_metrics = ['mean_squared_error', 'mean_absolute_error', 'accuracy']
    str_labels = ['MSE', 'MAE', 'Accuracy']
    loss = 'mae'

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=5,
                                      restore_best_weights=True)]

    # set seed
    # tensorflow.random.set_seed(0)

    model = keras.Sequential()
    model.add(keras.Input(shape=(6,)))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(12, activation=None))
    model.add(keras.layers.Flatten())
    print(model.summary())

    # compile, fit and evaluate model
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=str_metrics)

    history = model.fit(input_train_norm,
                        output_train_norm,
                        epochs=epochs,
                        batch_size=batchsize,
                        validation_data=(input_val_norm, output_val_norm),
                        verbose=1,
                        callbacks=callbacks)
    # training history; prepare pd.DataFrame for plotting later
    tr_history = pd.DataFrame(history.history)
    tr_history['epoch'] = history.epoch

    evaluation = model.evaluate(input_test_norm,
                                output_test_norm,
                                batch_size=batchsize,
                                callbacks=callbacks,
                                return_dict=False)
    print(evaluation)

    # Make predictions for later evaluation
    train_predictions_norm, val_predictions_norm, test_predictions_norm = [
        model.predict(subset) for subset in [input_train_norm,
                                             input_val_norm,
                                             input_test_norm]]
    train_predictions, val_predictions, test_predictions = [
        scaler_out.inverse_transform(subset)
        for subset in [train_predictions_norm,
                       val_predictions_norm,
                       test_predictions_norm]]

    # construct an output dict
    dln_data = {'input_train': input_train,
                'input_test': input_test,
                'input_val': input_val,
                'output_train': output_train,
                'output_test': output_test,
                'output_val': output_val,
                'input_train_norm': input_train_norm,
                'input_test_norm': input_test_norm,
                'input_val_norm': input_val_norm,
                'output_train_norm': output_train_norm,
                'output_test_norm': output_test_norm,
                'output_val_norm': output_val_norm,
                'test_predictions': test_predictions,
                'val_predictions': val_predictions,
                'train_predictions': train_predictions,
                'str_metrics': str_metrics,
                'str_labels': str_labels}

    # save the data, the trained model, scalers, evaluation metrics and history
    joblib.dump(scaler_in, path_out + os.sep + 'scaler_in_12x1.pkl')
    joblib.dump(scaler_out, path_out + os.sep + 'scaler_out_12x1.pkl')
    np.savez_compressed(path_out + os.sep + 'dln_data_12x1', **dln_data)
    model.save(path_out + os.sep + 'dln_model_12x1.h5')
    model.save(path_out + os.sep + 'dln_model_12x1.keras')
    with open(path_out + os.sep + 'tr_history_12x1.pickle', 'wb') as f:
        pickle.dump(tr_history, f)
    # save the information about for later reference
    np.savetxt(path_out + os.sep + 'layer_depths_12x1.txt',
               depth_layers,
               comments='')
    np.savetxt(path_out + os.sep + 'evaluation_metrics.txt',
               evaluation[1:],
               delimiter=',',
               header=(','.join(str_metrics)),
               comments='')

    return [model, ECa, scalers], [tr_history, evaluation], depth_layers


def main():

    FWD_MODEL = 'CS'
    NMODELS = 25
    ENSEMBLE_12x1 = []
    print('runnin branch')

    CLIMITS1 = [1, 80]
    CLIMITS2 = [1, 80]

    METRICS_AND_LABELS = [['mean_squared_error', 'MSE'],
                          ['mean_absolute_error', 'MAE'],
                          ['accuracy', 'Accuracy']]

    # define paths to training data and results
    # Subnet 1: 4 layers
    PATH_IN_4x1 = "../data/dl/training_data/1.5m_4layers/"
    PATH_OUT_4x1 = "../data/dl/models/1.5m_4layers/"
    EC4x1_train = np.load(PATH_IN_4x1 + os.sep + 'EC.npy',
                          allow_pickle=True)

    # Subnet 2: 6 layers
    PATH_IN_6x1 = "../data/dl/training_data/1.5m_6layers/"
    PATH_OUT_6x1 = "../data/dl/models/1.5m_6layers/"
    EC6x1_train = np.load(PATH_IN_6x1 + os.sep + 'EC.npy',
                          allow_pickle=True)

    # Subnet 3: 12 layers
    PATH_IN_12x1 = "../data/dl/training_data/1.5m_12layers/"
    PATH_OUT_12x1 = "../data/dl/models/1.5m_12layers/"
    EC12x1_train = np.load(PATH_IN_12x1 + os.sep + 'EC.npy',
                           allow_pickle=True)

    for member in range(0, NMODELS):

        # train the first subnet: 4x1
        model_and_data4x1, hist_eval4x1, depth4x1 = trainSubnet4x1(
            PATH_IN_4x1,
            PATH_OUT_4x1)
        model4x1, ECa4x1_input, scalers4x1 = model_and_data4x1
        tr_history4x1, evaluation4x1 = hist_eval4x1

        # make predictions
        EC4x1_predicted = performModelPrediction(ECa4x1_input.T,
                                                 model4x1,
                                                 scalers4x1[0],
                                                 scalers4x1[1])

        ECa4x1_VCP_fwdr, ECa4x1_HCP_fwdr = computeForwardModel(
            EC4x1_predicted, depth4x1,
            forward_model=FWD_MODEL)

        # train the second subnet: 6x1
        model_and_data6x1, hist_eval6x1, depth6x1 = trainSubnet6x1(
            PATH_IN_6x1,
            PATH_OUT_6x1,
            np.hstack([ECa4x1_VCP_fwdr, ECa4x1_HCP_fwdr]))
        model6x1, ECa6x1_input, scalers6x1 = model_and_data6x1
        tr_history6x1, evaluation6x1 = hist_eval6x1

        # make predictions
        EC6x1_predicted = performModelPrediction(ECa6x1_input.T,
                                                 model6x1,
                                                 scalers6x1[0],
                                                 scalers6x1[1])

        ECa6x1_VCP_fwdr, ECa6x1_HCP_fwdr = computeForwardModel(
            EC6x1_predicted, depth6x1,
            forward_model=FWD_MODEL)

        # train the third subnet: 12x1
        model_and_data12x1, hist_eval12x1, depth12x1 = trainSubnet12x1(
            PATH_IN_12x1,
            PATH_OUT_12x1,
            np.hstack([ECa6x1_VCP_fwdr, ECa6x1_HCP_fwdr]))
        model12x1, ECa12x1_input, scalers12x1 = model_and_data12x1
        tr_history12x1, evaluation12x1 = hist_eval12x1

        ENSEMBLE_12x1.append(model_and_data12x1)

    ENSEMBLE_PRED = []
    # make predictions for ensemble
    for member in ENSEMBLE_12x1:

        model12x1, ECa12x1_input, scalers12x1 = member
        EC12x1_predicted = performModelPrediction(ECa12x1_input.T,
                                                  model12x1,
                                                  scalers12x1[0],
                                                  scalers12x1[1])
        ENSEMBLE_PRED.append(EC12x1_predicted)

    return ENSEMBLE_PRED
    # # plotting starts here for all nets
    # hist_list = [tr_history4x1, tr_history6x1, tr_history12x1]
    # eval_list = [evaluation4x1, evaluation6x1, evaluation12x1]
    # EC_predicted_list = [EC4x1_predicted, EC6x1_predicted,
    #                      EC12x1_predicted]
    # EC_train_list = [EC4x1_train, EC6x1_train, EC12x1_train]
    # depths_list = [depth4x1, depth6x1, depth12x1]
    # paths_out_list = [PATH_OUT_4x1, PATH_OUT_6x1, PATH_OUT_12x1]
    # fig_title_list = [
    #     ["EC models used as input in subnet 1",
    #      "Predicted EC models for subnet 1"],
    #     ["EC models used as input in subnet 2",
    #      "Predicted EC models for subnet 2"],
    #     ["EC models used as input in subnet 3",
    #      "Predicted EC models for subnet 3"]]

    # for fidx, path in enumerate(paths_out_list):
    #     print("Plotting the results for: {}".format(path))
    #     plotTrainingEvolution(hist_list[fidx],
    #                           eval_list[fidx],
    #                           path,
    #                           METRICS_AND_LABELS)
    #     plotAll1DModels(EC_train_list[fidx], depths_list[fidx],
    #                     "EC [mS/m]",
    #                     fig_title_list[fidx][0],
    #                     CLIMITS1,
    #                     path + os.sep + '02_InputEC.png')
    #     plotAll1DModels(EC_predicted_list[fidx], depths_list[fidx],
    #                     "Predicted EC [mS/m]",
    #                     fig_title_list[fidx][1],
    #                     CLIMITS1,
    #                     path + os.sep + '03_PredictedEC.png')
    #     plotRegressionsTrainingAndPredictedEC(
    #         EC_predicted_list[fidx],
    #         EC_train_list[fidx],
    #         depths_list[fidx],
    #         limits=CLIMITS2,
    #         path_plot=path + os.sep + '04_Regression.png')


if __name__ == "__main__":
    ensemble = main()
