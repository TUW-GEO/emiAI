# -*- coding: utf-8 -*-
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from aggregator_utils import joinAllSections
from downsample_utils import computeLayerMids, downsampleCompositeModel
from generator_utils import (computeAverageModel, generateNewModels,
                             loadAndConvertResIPyMesh, performRandomExtraction,
                             write2logfile)
from plotting_utils import indicateExtractionAreas, plotModelResIPy


def main():

    desc_text = ("Pseudo-random 1D model generation based on ERT data. Used to"
                 + " generate training data for a deep-learning network.")
    parser = argparse.ArgumentParser(description=desc_text)

    # Add a positional argument
    parser.add_argument('settings_json', type=str,
                        help="Path to the settings JSON")

    # Add an optional keyword argument
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Plot figures during model generation")
    parser.add_argument('--aggregate', action=argparse.BooleanOptionalAction,
                        default=False,
                        help=("Aggregate the generated models for each profile"
                              + " into a composite model."))
    parser.add_argument('--aggregate_only',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=("Do only the aggregation for the in the config "
                              + "specified profiles. No new model generation"
                              + "is performed."))
    parser.add_argument('--downsample', action=argparse.BooleanOptionalAction,
                        default=False,
                        help=("Downsample the 1D models from composite model "
                              + "to composite depth models with 6 and 4 layers"
                              + "  as specified in config.json. Only applies"
                              + " if --aggregate is specified."))

    # Parse the arguments
    args = parser.parse_args()
    plot = args.plot
    aggregate = args.aggregate
    aggregate_only = args.aggregate_only
    downsample = args.downsample

    if downsample and (aggregate is False) and (aggregate_only is False):
        print("Downsampling only possible if aggregation is specified!")
        downsample = False

    if args.settings_json is not None:
        try:
            settings = json.load(open(args.settings_json, 'r'))
        except FileNotFoundError as error:
            print("An exception occurred:", type(error).__name__, "–", error)
            print("config.json not found. Filepath is probably wrong.")
            print("Trying default path.")
            try:
                settings = json.load(open("./config.json", 'r'))
            except FileNotFoundError as error:
                print("An exception occurred:",
                      type(error).__name__, "–",
                      error)
                print("config.json not found. Stopping execution.")
                return None

    if not os.path.exists(settings["PATH_OUT"]):
        os.makedirs(settings["PATH_OUT"])

    path_res = (settings["PATH_OUT"]
                + str(settings["DEPTH"]) + 'm'
                + '_' + str(settings["NLAYERS"]) + 'layers' + os.sep)

    if aggregate_only is False:

        for profile in settings["FID_LIST"]:

            # path_temp = settings["PATH_OUT"] + profile + os.sep
            path_temp = (path_res
                         + profile + os.sep)
            print("Processing profile: {}".format(profile))
            print("Writing results to: {}".format(path_temp))

            if plot:
                path_fig = path_temp
            else:
                path_fig = None

            # generate folders if not existant
            if not os.path.exists(path_temp):
                os.makedirs(path_temp)

            # loading data
            print('Load inversion results and sensor array.')
            sensors = np.genfromtxt((settings["PATH_DATA"]
                                     + profile + '_grid.csv'),
                                    skip_header=1, delimiter='\t')
            mesh, _, xyz = loadAndConvertResIPyMesh(
                (settings["PATH_DATA"]
                 + profile
                 + os.sep + profile),
                value_id="Magnitude(ohm.m)")

            # compute the extraction width as the mean
            # sensor distance * scaling option
            extraction_width = np.mean(np.diff(
                sensors[:, 0]))*settings["EXTRACTION_WIDTH_SCALER"]
            print("Extraction width used: {:.2f}".format(extraction_width))

            # extract the data and the corresponding cellcenters from the ERT
            # data
            print("Performing the random extraction.")
            extracted_data = performRandomExtraction(
                xyz,
                sensors,
                settings["DEPTH"],
                extraction_width,
                nextraction_range=settings["NEXTRACTION_RANGE"],
                side_skip=settings["SIDE_SKIP"])

            if plot is True:
                fig, ax = plt.subplots(nrows=1,
                                       ncols=1,
                                       figsize=(7.2, 5.2))
                ax = plotModelResIPy(ax,
                                     mesh,
                                     "Magnitude(ohm.m)",
                                     settings["ER_PLOT_RANGE"],
                                     sensors=sensors)
                ax.set_title(profile, pad=20)

                # plot the extraction areas
                [indicateExtractionAreas(ax, idx, extracted_data[idx])
                 for idx in range(len(extracted_data))]

                fig.savefig(path_fig + 'section_extraction_points.png',
                            dpi=200,
                            bbox_inches='tight')
                plt.close(fig)

            # compute the average models for each extraction position
            print("## Computing the average models from extracted data. ##")
            average_models = [computeAverageModel(
                extracted_data[idx],
                settings["LAYER_EDGES"],
                path_fig=path_fig,
                n0_offset=settings["N0_OFFSET"],
                total_offset=settings["TOTAL_OFFSET"])
                for idx in range(len(extracted_data))]

            avm = average_models

            # now compute the new 1D models for each extraction position
            print("## Generating new model values now. ##")
            generated_models = [generateNewModels(
                settings["LAYER_EDGES"],  # layer edges
                [avm[idx][0], avm[idx][1]],  # bin statistic = mean + bin std
                settings["NMODELS"],
                bounds=settings["BOUNDS"],
                exag_fac=settings["EXAG_FACTORS"],
                reverse=settings["REVERSE"],
                n0_scaler=settings["N0_SCALER"],
                nend_scaler=settings["NEND_SCALER"],
                restrict_nend=settings["RESTRICT_NEND"],
                layer_smoothing=settings["SMOOTHER"],
                est_distribution=settings["DISTRIBUTION"],
                path_fig=path_fig) for idx in range(len(average_models))]

            xmid_extraction_points = [extracted_data[idx][0]
                                      for idx in range(len(extracted_data))]
            extracted_xz_coords = [extracted_data[idx][1]
                                   for idx in range(len(extracted_data))]
            extracted_rho = [extracted_data[idx][2]
                             for idx in range(len(extracted_data))]

            binned_data_depth = settings["LAYER_EDGES"]
            binned_rho = [average_models[idx][0]
                          for idx in range(len(average_models))]
            binned_std = [average_models[idx][1]
                          for idx in range(len(average_models))]

            generated_data_depths = generated_models[0][0]
            generated_step_models = [generated_models[idx][1]
                                     for idx in range(len(generated_models))]
            generated_layer_values = [generated_models[idx][2]
                                      for idx in range(len(generated_models))]
            training_data = {'xmid_extraction_points': xmid_extraction_points,
                             'extracted_xz_coords': extracted_xz_coords,
                             'extracted_rho': extracted_rho,
                             'binned_data_depth': binned_data_depth,
                             'binned_rho': binned_rho,
                             'binned_std': binned_std,
                             'generated_data_depths': generated_data_depths,
                             'generated_step_models': generated_step_models,
                             'generated_layer_values': generated_layer_values}
            np.savez_compressed(path_temp + profile
                                + '_' + str(settings["DEPTH"]) + 'm'
                                + '_' + str(settings["NLAYERS"]) + 'layers',
                                **training_data)

            print("Finished processing profile: {}".format(profile))

            # also write a log file with settings
            write2logfile(settings,
                          path_temp,
                          profile,
                          extraction_width,
                          extracted_data,  # for extraction location
                          )

        if aggregate:
            print("Aggregating individual profiles into composite data set.")
            path_list = [path_res
                         + profile + os.sep
                         + profile
                         + '_' + str(settings["DEPTH"]) + 'm'
                         + '_' + str(settings["NLAYERS"]) + 'layers.npz'
                         for profile in settings["FID_LIST"]]
            comp, comp_depths = joinAllSections(path_list)
            composite_training_data = {'composite_model_depth': comp_depths,
                                       'composite_model': comp}
            np.savez_compressed(path_res
                                + 'composite_model', **composite_training_data)

            if downsample:
                # start with 6 layer model
                composite_model_6l = downsampleCompositeModel(
                    comp,
                    computeLayerMids(comp_depths),
                    computeLayerMids(settings["LAYER_EDGES_6"])
                )
                composite_training_data_6l = {
                    'composite_model_depth': settings["LAYER_EDGES_6"],
                    'composite_model': composite_model_6l}
                path_6l = (settings["PATH_OUT"]
                           + str(settings["DEPTH"]) + 'm'
                           + '_6layers' + os.sep)
                # generate folders if not existant
                if not os.path.exists(path_6l):
                    os.makedirs(path_6l)
                np.savez_compressed(path_6l
                                    + 'composite_model',
                                    **composite_training_data_6l)

                # continue with 4 layer model
                composite_model_4l = downsampleCompositeModel(
                    comp,
                    computeLayerMids(comp_depths),
                    computeLayerMids(settings["LAYER_EDGES_4"])
                )
                composite_training_data_4l = {
                    'composite_model_depth': settings["LAYER_EDGES_4"],
                    'composite_model': composite_model_4l}
                path_4l = (settings["PATH_OUT"]
                           + str(settings["DEPTH"]) + 'm'
                           + '_4layers' + os.sep)
                # generate folders if not existant
                if not os.path.exists(path_4l):
                    os.makedirs(path_4l)
                np.savez_compressed(path_4l
                                    + 'composite_model',
                                    **composite_training_data_4l)

    else:

        print("Aggregating individual profiles into composite data set.")
        path_list = [path_res
                     + profile + os.sep
                     + profile
                     + '_' + str(settings["DEPTH"]) + 'm'
                     + '_' + str(settings["NLAYERS"]) + 'layers.npz'
                     for profile in settings["FID_LIST"]]
        comp, comp_depths = joinAllSections(path_list)
        composite_training_data = {'composite_model_depth': comp_depths,
                                   'composite_model': comp}
        np.savez_compressed(path_res
                            + 'composite_model', **composite_training_data)

        if downsample:
            # start with 6 layer model
            composite_model_6l = downsampleCompositeModel(
                comp,
                computeLayerMids(comp_depths),
                computeLayerMids(settings["LAYER_EDGES_6"])
            )
            composite_training_data_6l = {
                'composite_model_depth': settings["LAYER_EDGES_6"],
                'composite_model': composite_model_6l}
            path_6l = (settings["PATH_OUT"]
                       + str(settings["DEPTH"]) + 'm'
                       + '_6layers' + os.sep)
            # generate folders if not existant
            if not os.path.exists(path_6l):
                os.makedirs(path_6l)
            np.savez_compressed(path_6l
                                + 'composite_model',
                                **composite_training_data_6l)

            # continue with 4 layer model
            composite_model_4l = downsampleCompositeModel(
                comp,
                computeLayerMids(comp_depths),
                computeLayerMids(settings["LAYER_EDGES_4"])
            )
            composite_training_data_4l = {
                'composite_model_depth': settings["LAYER_EDGES_4"],
                'composite_model': composite_model_4l}
            path_4l = (settings["PATH_OUT"]
                       + str(settings["DEPTH"]) + 'm'
                       + '_4layers' + os.sep)
            # generate folders if not existant
            if not os.path.exists(path_4l):
                os.makedirs(path_4l)
            np.savez_compressed(path_4l
                                + 'composite_model',
                                **composite_training_data_4l)


if __name__ == "__main__":
    main()
