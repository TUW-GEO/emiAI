# -*- coding: utf-8 -*-
import glob
import json
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import resipy as rsp
from scipy import stats
from scipy.signal import savgol_filter as svg
from scipy.stats import truncnorm

from plotting_utils import (plotAverage1DModelAndScatter, plotStepModel,
                            prepare1DStepmodel)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """Defines a truncated normal distribution to draw samples from.

    Args:
        mean (int, optional): Distribution mean. Defaults to 0.
        sd (int, optional): Distribution standard deviation. Defaults to 1.
        low (int, optional): Lower bound of distribution. Defaults to 0.
        upp (int, optional): Upper bound of distribution. Defaults to 10.

    Returns:
        scipy.stats.rv_continous: Truncated normal distribution to draw
                                  samples from.
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def convolutionSmooth(array, npoints):
    """Moving average filter with edge handling based on np.convolve. 

    Args:
        array (np.array): Array with values that should be "smoothed".
        npoints (int): Number of points to compute the average on. Needs to be 
                       an odd number. 

    Returns:
        np.array: Returns array after application of smoothing filter.
    """
    if npoints % 2 == 0:
        npoints = npoints + 1
    return (np.convolve(array, np.ones(npoints, dtype='float'), 'same')
            / np.convolve(np.ones(len(array)), np.ones(npoints), 'same'))


def smoothLayerValues(layer_values, options_dict):
    """Applies a smoothing filter to the provided 1D model values.
    
    Args:
        layer_values (np.array): Contains the values of the 1D model.
        options_dict (dict): Dictionary containing the type of smoothing filter
                             and the associated parameters.

    Returns:
        smoothed_values (np.array): Array after application of
                                    smoothing filter.
    """
    if options_dict['smoother'] == 'savgol':
        print('Using a Savitzky-Golay filter with'
              + ' {} points'.format(options_dict['npoints'])
              + ' and order of {}'.format(options_dict['order']))
        smoothed_values = svg(layer_values,
                              options_dict['npoints'],
                              options_dict['order'])
    elif options_dict['smoother'] == 'convolve':
        print('Using a moving average filter with'
              + ' {} points'.format(options_dict['npoints']))
        smoothed_values = convolutionSmooth(layer_values,
                                            options_dict['npoints'])

    return smoothed_values

    
def loadAndConvertResIPyMesh(path_data, value_id="Magnitude(ohm.m)"):
    """Loads inversion results of ResIPy from a vtk file and converts the
    mesh to a .csv containing the x y z positions of the mesh nodes and
    the values (-> value_id) associated to it. The .csv is read then again and
    are passed back as xyz (containing the x y z positions of the mesh nodes)
    and data (vector of the inverted electrical property associated with it)
    along with the ResIPy mesh.
    
    Args:
        path_data (str): Path to inversion results in vtk format.
        value_id (str, optional): Which property to read from the vtk.
                                  Defaults to "Magnitude(ohm.m)".

    Returns:
        mesh, data, xyz: (resipy.mesh, np.array, np.array)
    """

    mesh = rsp.meshTools.vtk_import(path_data + '.vtk')
    mesh.toCSV(file_name=path_data + '_mesh.csv')
    df = pd.read_csv(path_data + '_mesh.csv')
    data = df[value_id].values
    xyz = df.loc[:, ['x', 'z', value_id]].values
    return mesh, data, xyz


def getRandomExtractionPositions(xvalues, nextractions, extr_width):
    """For x coordinates of a mesh do a random sampling of the x values for
    a specified number of extractions. Check if the distance of neighboring
    extraction points is larger than the extraction width.

    Args:
        xvalues (np.array): Array containing x coordinates of a mesh.
        nextractions (int): Number of extraction points.
        extr_width (float): Width of the extraction box.
        
    Returns:
        xmids (np.array): Contains the x coordinates (along the mesh) of the
                        extraction locations.
    """
    t = datetime.now()
    np.random.seed(int(time.mktime(t.timetuple())))
    xmids = []
    for ii in range(1000):
        print('Random sampling extraction locations: {}'.format(ii+1))
        xmids = np.random.choice(xvalues, nextractions, replace=False)
        xmids_sort = np.sort(xmids)
        xm_diff = np.diff(xmids_sort)
        xm_diff2 = np.ones(len(xmids))*10
        xm_diff2[1:] = xm_diff
        if len(xmids_sort[xm_diff2 < extr_width]) == 0:
            break  
        
    return np.array(xmids)
        

def extractValues(cellcenters, data, left, right, bottom, top=None):
    """Extract x z values and corresponding data from mesh cellcenters or nodes
    within a defineable bounding box (left, right, bottom, top).

    Args:
        cellcenters (np.array): Array containing x z positions of mesh
                                cellcenters or nodes.
        data (np.array): Data corresponding to each mesh cellcenter or node.
        left (float): Left boundary of extraction box in x coordinates.
        right (float): Right boundary of extraction box in x coordinates.
        bottom (float): Lower boundary of extraction box in z coordinates.
        top (float, optional): Upper boundary of extraction box
                               in z coordinates. Defaults to None.

    Returns:
        xz_ex (np.array): x z cellcenter or nodes coordinates within
                          bounding box.
        ex (np.array): Data corresponding to x z values.
    """
    if top is None:
        ex = data[np.where(((cellcenters[:, 0] > left)
                            & (cellcenters[:, 0] < right)
                            & (cellcenters[:, 1] > bottom)))]
        xz_ex = cellcenters[np.where(((cellcenters[:, 0] > left)
                                      & (cellcenters[:, 0] < right)
                                      & (cellcenters[:, 1] > bottom)))]
        return xz_ex, ex
    else:
        ex = data[np.where(((cellcenters[:, 0] > left)
                            & (cellcenters[:, 0] < right)
                            & (cellcenters[:, 1] < top)
                            & (cellcenters[:, 1] > bottom)))]
        xz_ex = cellcenters[np.where(((cellcenters[:, 0] > left)
                                      & (cellcenters[:, 0] < right)
                                      & (cellcenters[:, 1] < top)
                                      & (cellcenters[:, 1] > bottom)))]
        return xz_ex, ex


def performRandomExtraction(cellcenters, sensors, depth, extr_width,
                            nextraction_range=[7, 12],
                            side_skip=4):
    """Perform random extraction with bounding boxes for a specified number
    of times. Check that extraction positions do not overlap and are not in low
    sensitive areas (-> side_skip).

    Args:
        cellcenters (np.array): Array containing x z positions of mesh
                                cellcenters or nodes.
        sensors (np.array): Contains x z position of electrodes.
        depth (float): Maximum depth for extraction. Bottom of bounding box.
        extr_width (float): Width of bounding box.
        nextraction_range (list, optional): Range for the number of extraction
                                            locations. Defaults to [7, 12].
        side_skip (int, optional): Defines the low sensitivity zone at the
                                   boundaries of the model. Refers to the
                                   number of sensors on each side.
                                   Defaults to 4.

    Returns:
        extracted_data (list): Contains the following:
                               Index 0: xmid -> X position along the profile
                               where data has been extracted
                               Index 1: cellcenters_extracted -> x, z pos of
                               the extracted nodes inside the box
                               Index 2: data_extracted -> data corresponding to
                               nodes
    """
    
    # check if provided number of extraction points is correctly ordered
    if nextraction_range[0] > nextraction_range[1]:
        print('Upper extraction point number is larger than lower number!'
              'Defaulting back to [7, 12]!')
        nextraction_range = [7, 12]

    # now get the number of actual extraction points from range
    nextractions = random.randint(nextraction_range[0], nextraction_range[1])
    xvalues = cellcenters[:, 0].copy()

    # for the number of extractions get random positions along the profile
    # check that random positions do not overlap
    xmids = getRandomExtractionPositions(xvalues,
                                         nextractions,
                                         extr_width)
    
    xmids_orig = xmids.copy()
    # perform a check if extraction location is in low sensitvity area
    # defined by side_skip, which refers to the number of sensors on each side
    # left side
    xmids = xmids[xmids >= sensors[side_skip, 0]]
    # right side
    xmids = xmids[xmids <= sensors[-side_skip, 0]]
    dropped_values = np.setdiff1d(xmids_orig, xmids)
    if len(dropped_values) != 0:
        for elem in dropped_values:
            print('Skipping xmid='
                  + '{:.2f} m -> in low sensitivity area!'.format(elem))

    extracted_data = []
    for extraction_num, xmid in enumerate(xmids):
        print('Position {} at xmid={:.1f} m'.format(extraction_num + 1, xmid))
        
        idx = (np.abs(sensors[:, 0] - xmid)).argmin()
        zvalue_at_xmid = sensors[idx, 1]

        bottom = depth
        cellcenters_extracted, data_extracted = extractValues(
            cellcenters,
            cellcenters[:, -1],  # contains data to x z values
            xmid - extr_width / 2,
            xmid + extr_width / 2,
            bottom,
            top=zvalue_at_xmid)
        extracted_data.append([xmid, cellcenters_extracted, data_extracted])

    return extracted_data


def preprocessAverageModel(bin_vals, layer_smoothing):
    """Preprocessing of average model. This contains gap filling with
    median values of model values und standard deviations as well as
    upscaling of very low standard deviations to median value.
    If specified, the average model is also smoothed.

    Args:
        bin_vals (list): Contains two np.arrays with the model values and
                            the associated standard deviation in each layer.
        layer_smoothing (dict): Dictionary containing the type of smoothing
                                filter and the associated parameters.

    Returns:
        bin_values, bin_std: Bin values and standard deviations after
                                preprocessing and possible smoothing.
    """
    
    bin_values, bin_std = bin_vals
    # empty bins are set to median value of model
    bin_values[np.isnan(bin_values)] = np.nanmedian(bin_values)
    bin_std[np.isnan(bin_std)] = np.nanmedian(bin_std)
    # very low bin stds are scaled to median value
    bin_std[bin_std < 0.1] = np.nanmedian(bin_std)
    
    if layer_smoothing is not None:
        print('Applied layer smoothing to average model!')
        bin_values = smoothLayerValues(bin_values, layer_smoothing)

    return bin_values, bin_std


def computeModelChangesVector(bin_values, restrict_nend=False):
    """Compute a vector that indicates in which direction the model is
    changing for increasing depth. Used for generation of new model values.
    The "direction", i.e., if the model is increasing or decreasing is
    determined by a sign change of neighboring values.
    
    Args:
        bin_values (np.array): Values of the 1D model.
        restrict_nend (bool, optional): Restrict direction changes for last
                                        layer to the direction of the one
                                        above. Defaults to False.

    Returns:
        filter_vec (np.array): Vector that indicates in which direction the
                                model is changing. Can be quantified by
                                checking sign changes of neighboring values.
    """
    filter_vec = np.zeros((len(bin_values)))
    filter_vec[0:-1] = np.diff(bin_values)
    
    if restrict_nend is True:
        filter_vec[-1] = filter_vec[-2]
        
    return filter_vec


def generateNewModels(layer_edges, bin_vals,
                      nmodels,
                      bounds=[0.01, 1000],
                      exag_fac=[5, 5],
                      reverse=True,
                      n0_scaler=0.35,
                      nend_scaler=1,
                      restrict_nend=False,
                      layer_smoothing=None,
                      est_distribution="normal",
                      path_fig=None,
                      **kwargs):
    """Pseudo-randomly generate new 1D EC models from the given average model
    within specified bounds. Scaled bin stds are used to define ranges in which
    the new models may vary. Hard caps can be specified as well as the specific
    scaling options for the first and last bin/layer. Optional smoothing of the
    obtained models is possible.
    New model values can be drawn from a uniform or truncated normal
    distribution.

    Args:
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        bin_vals (list): Contains two np.arrays with the model values and
                            the associated standard deviation in each layer.
        nmodels (int): Number of new models to be generated.
        bounds (list, optional): Maximum new model values in each layer. If
                                 modelled values exceed bounds they are scaled
                                 down to fit into upper or lower bound values.
                                 Bound values in Ohmm.
                                 Defaults to [0.01, 1000].
        exag_fac (list, optional): Scaling factor for all bin stds.
                                   Defaults to [5, 5].
        reverse (bool, optional): Inverts the model change direction vector and
                                  permits a larger number of models. If True
                                  doubles the number of nmodels.
                                  Defaults to True.
        n0_scaler (float, optional): Scaling factor for first bin std.
                                     Defaults to 0.35.
        nend_scaler (int, optional): Scaling factor for last bin std.
                                     Defaults to 1.
        restrict_nend (bool, optional): If True does not permit a model change
                                        direction for last bin/layer and uses
                                        the direction from the one above.
                                        Defaults to False.
        layer_smoothing (dict, optional): Dictionary containing the type of
                                          smoothing filter and the associated
                                          parameters. Defaults to None.
        est_distribution (str, optional): Distribution to draw the random
                                          samples from. Can be "uniform" and
                                          "normal". Defaults to "normal".
        path_fig (_type_, optional): Path to folder where figures should be
                                     saved. If None no figures are save.
                                     Defaults to None.
        xlabel (str, optional): Custom x label for plot.
                                Defaults to r"$\rho$ [$\Omega$m]".
        ylabel (str, optional): Custom y label for plot.
                                Defaults to "Depth [m]".

    Returns:
        profile (np.array): Depths of the 1D step model
        output_stepmodels (np.array): Array that contains the step models for
                                      the preprocessed average model and the
                                      new generated models.
        output_layer_values (np.array): Array that contains the values of
                                        the preprocessed average model and
                                        newly generated models.
    """

    # preprocess the average model: gap filling and scaling
    bin_values, bin_std = preprocessAverageModel(
        bin_vals,
        layer_smoothing=layer_smoothing)
          
    # if specified scale specific standard deviations, first and last
    bin_std[0] = bin_std[0] * n0_scaler
    bin_std[-1] = bin_std[-1] * nend_scaler

    # prepare the stepmodels
    profile, rhop = prepare1DStepmodel(layer_edges, bin_values)
    _, stdp = prepare1DStepmodel(layer_edges, bin_std)

    # compute the model changes direction vector; needed to indicate in which
    # direction new models may change
    filter_vec = computeModelChangesVector(bin_values,
                                           restrict_nend=restrict_nend)

    # compute the bounds in which the new models may vary
    # here the exag_fac scales the standard deviations
    # model values + scaled standard deviations
    stdp_plus = bin_values + exag_fac[1] * bin_std
    stdp_minus = bin_values - exag_fac[0] * bin_std

    # reverse = True -> filter_vec direction is inverted to allow more models
    # doubles the number of output models specified by nmodels!
    if reverse is True:
        output_stepmodels = np.empty((len(rhop), (nmodels * 2) + 1))
        output_layer_values = np.empty((len(bin_values), (nmodels * 2) + 1))
    else:
        output_stepmodels = np.empty((len(rhop), nmodels + 1))
        output_layer_values = np.empty((len(bin_values), nmodels + 1))

    # 1st model in output is the preprocessed average model
    output_stepmodels[:, 0] = rhop
    output_layer_values[:, 0] = rhop[0::2]
    
    if path_fig is not None:
        fig, ax = plt.subplots(figsize=(3.6, 5.5))
        ax = plotStepModel(ax, profile, rhop, linestyle='-',
                           linewidth=1.9, zorder=10)
        # plot bounds at which restriction to model values happen
        ax.plot([bounds[0], bounds[0]], [
                np.min(layer_edges), -np.max(layer_edges)],
                '-', color='k', linewidth=1.9)

        # step model values + scaled standard deviations
        stdp_plus_step = rhop + exag_fac[1] * stdp
        stdp_minus_step = rhop - exag_fac[0] * stdp
        
        # left std bound
        ax = plotStepModel(ax, profile, stdp_minus_step,
                           color='Grey',
                           linestyle='-',
                           linewidth=0.9)
        # right std bound
        ax = plotStepModel(ax, profile, stdp_plus_step,
                           color='Grey',
                           linestyle='-',
                           linewidth=0.9)

        ax.set_xlabel(kwargs.get('xlabel', r"$\rho$ [$\Omega$m]"))
        ax.set_ylabel(kwargs.get('ylabel', 'Depth [m]'))

    # now iterate over the number of requested models and check if reverse
    # if so do the reverse model too
    model_count = 1
    for _ in range(0, nmodels):

        # define a function which takes the distribution and if reverse
        rhop_new = generateNewLayerValues(
            bin_values,
            filter_vec,
            [stdp_plus, stdp_minus, bin_std],
            bounds,
            reverse=False,
            distribution=est_distribution,
            exag_fac=exag_fac)

        # check if the generated model should be smoothed
        if ((layer_smoothing is not None)
            and ("apply_all" in layer_smoothing)
                and (layer_smoothing['apply_all'] is True)):
            rhop_new = smoothLayerValues(rhop_new, layer_smoothing)

        _, rhop_new_step = prepare1DStepmodel(layer_edges, rhop_new)

        if path_fig is not None:
            ax = plotStepModel(ax, profile, rhop_new_step,
                               color='r', linestyle='--', linewidth=0.1)
        # store the new model in array and increase model count
        output_stepmodels[:, model_count] = rhop_new_step
        output_layer_values[:, model_count] = rhop_new_step[0::2]
        model_count += 1

        if reverse is True:

            rhop_new = generateNewLayerValues(
                bin_values,
                filter_vec,
                [stdp_plus, stdp_minus, bin_std],
                bounds,
                reverse=True,
                distribution=est_distribution,
                exag_fac=exag_fac)

            # check if the generated model should be smoothed
            if ((layer_smoothing is not None)
                and ("apply_all" in layer_smoothing)
                    and (layer_smoothing['apply_all'] is True)):
                rhop_new = smoothLayerValues(rhop_new, layer_smoothing)

            _, rhop_new_step = prepare1DStepmodel(layer_edges, rhop_new)
            
            if path_fig is not None:
                ax = plotStepModel(ax, profile, rhop_new_step,
                                   color='c', linestyle='--', linewidth=0.1)
            # store the new model in array and increase model count
            output_stepmodels[:, model_count] = rhop_new_step
            output_layer_values[:, model_count] = rhop_new_step[0::2]
            model_count += 1

    if path_fig is not None:
        # check if a first file exists
        if os.path.exists(path_fig + 'modelled01.png'):
            # now check the current number
            files_there = glob.glob(path_fig + 'modelled*.png')
            max_num = max([int(file.split(os.sep)[-1].split('.')[-2][-2:])
                           for file in files_there])
            fig.savefig((path_fig + 'modelled'
                        + str(max_num + 1).zfill(2) + '.png'),
                        dpi=200,
                        bbox_inches='tight')
        else:
            fig.savefig(path_fig + 'modelled01.png',
                        dpi=200,
                        bbox_inches='tight')

    return profile, output_stepmodels, output_layer_values


def generateNewLayerValues(bin_val,
                           filt_vec,
                           std_bounds,
                           bounds,
                           reverse=True,
                           distribution="normal",
                           exag_fac=[5, 5]):
    """Generates new layer values for given standard deviations,
    scaling factor, bounds and specified sampling distribution. Also the
    direction in which neighboring layer values change is taken into account.

    Args:
        bin_val (np.array): Preprocessed average model bin values.
        filt_vec (np.array): Array indicating in which direction neighboring
                             depth layers may vary.
        std_bounds (list): Contains the upper and lower bounds defined by the
                           standard deviations and the raw standard deviation
                           in each bin. Only used if distribution="uniform".
        bounds (list): Maximum new model values in each layer. If modelled
                       values exceed bounds they are scaled down to fit into
                       upper or lower bound values. Bound values in Ohmm.
        reverse (bool, optional): Inverts the model change direction vector.
                                  Defaults to True.
        distribution (str, optional): Distribution to draw the random samples
                                      from. Can be "uniform" and "normal".
                                      Defaults to "normal".
        exag_fac (list, optional): Scaling factor for all bin stds.
                                   Defaults to [5, 5].

    Returns:
        rhop_new (np.array): Newly generated 1D EC model.
    """
    stdp_plus, stdp_minus, bin_std = std_bounds

    if reverse is False:
        if distribution == 'uniform':
            rhop_new = bin_val.copy()
            # for positive direction change draw a sample between the
            # bin val = average model and stdp_plus bounds
            rhop_new[filt_vec > 0] = np.random.uniform(
                stdp_plus[filt_vec > 0], bin_val[filt_vec > 0])
            # for negative direction change draw a sample between the average
            # model and stdp_minus bounds
            rhop_new[filt_vec < 0] = np.random.uniform(
                stdp_minus[filt_vec < 0], bin_val[filt_vec < 0])
            # if no change use between stdp_minus and stdp_plus
            rhop_new[filt_vec == 0] = np.random.uniform(
                stdp_minus[filt_vec == 0], stdp_plus[filt_vec == 0])

            # for values exceeding the upper bounds draw sample restrict to
            # range within bounds
            rhop_new[rhop_new > bounds[1]] = np.random.uniform(
                bounds[1] * np.random.uniform(0.1, 0.5), bounds[1])

            rhop_new[rhop_new < bounds[0]] = np.random.uniform(
                bounds[0] * np.random.uniform(1.1, 10), bounds[0])

            return rhop_new

        elif distribution == 'normal':
            rhop_new = bin_val.copy()

            # check if positive changes
            if (len(bin_val[filt_vec > 0]) > 0):

                # get bin and std values with positive change
                bin_val_up = bin_val[filt_vec > 0]
                stdp_up = bin_std[filt_vec > 0]
 
                upscale = exag_fac[1]
                # check if upscaled std bounds exceed hard limits
                # if so harshly downscale the stds
                fvec2 = bin_val_up + stdp_up * upscale > bounds[1]
                stdp_up[fvec2] = (
                    (bounds[1] - bin_val_up[fvec2]) * 0.1)

                # array for new model values in positive direction
                normal_up = np.zeros(len(bin_val_up))

                # for each layer with positive change draw from truncated
                # normal distribution with mean = bin val = average model
                # sd = bin std * upscale; lower bound = bin val;
                # upper bound = bin_val + bin std * upscale
                for idx in range(0, len(bin_val_up)):
                    tr_norm = truncated_normal(
                        bin_val_up[idx],
                        stdp_up[idx] * upscale,
                        low=bin_val_up[idx],
                        upp=(bin_val_up[idx] + stdp_up[idx] * upscale))
                    normal_up[idx] = tr_norm.rvs(1)[0]
                rhop_new[filt_vec > 0] = normal_up

            # check if negative changes
            if (len(bin_val[filt_vec < 0]) > 0):

                bin_val_dw = bin_val[filt_vec < 0]
                stdp_dw = bin_std[filt_vec < 0]

                upscale = exag_fac[0]
                fvec3 = bin_val_dw - stdp_dw * upscale < bounds[0]
                stdp_dw[fvec3] = (
                    (bin_val_dw[fvec3] - bounds[0]) * 0.1)

                normal_dw = np.zeros(len(bin_val_dw))

                for idx in range(0, len(bin_val_dw)):

                    tr_norm = truncated_normal(
                        bin_val_dw[idx],
                        stdp_dw[idx] * upscale,
                        low=(bin_val_dw[idx] - stdp_dw[idx] * upscale),
                        upp=bin_val_dw[idx])

                    normal_dw[idx] = tr_norm.rvs(1)[0]
                rhop_new[filt_vec < 0] = normal_dw

            return rhop_new

    # reverse is True: filt_vec with switched sign
    else:
        if distribution == 'uniform':
            rhop_new = bin_val.copy()
            rhop_new[filt_vec > 0] = np.random.uniform(
                stdp_minus[filt_vec > 0], bin_val[filt_vec > 0])
            rhop_new[filt_vec < 0] = np.random.uniform(
                stdp_plus[filt_vec < 0], bin_val[filt_vec < 0])
            rhop_new[filt_vec == 0] = np.random.uniform(
                stdp_minus[filt_vec == 0], stdp_plus[filt_vec == 0])

            rhop_new[rhop_new > bounds[1]] = np.random.uniform(
                bounds[1] * np.random.uniform(0.1, 0.5),
                bin_val[rhop_new > bounds[1]])

            rhop_new[rhop_new < bounds[0]] = np.random.uniform(
                bounds[0] * np.random.uniform(1.5, 1.9),
                bin_val[rhop_new < bounds[0]])

            return rhop_new

        elif distribution == 'normal':

            rhop_new = bin_val.copy()

            if (len(bin_val[filt_vec < 0]) > 0):

                bin_val_up = bin_val[filt_vec < 0]
                stdp_up = bin_std[filt_vec < 0]

                upscale = exag_fac[1]
                std_fac = upscale
                fvec2 = bin_val_up + stdp_up * upscale > bounds[1]
                stdp_up[fvec2] = (
                    (bounds[1] - bin_val_up[fvec2]) * 0.1)

                normal_up = np.zeros(len(bin_val_up))

                for idx in range(0, len(bin_val_up)):

                    tr_norm = truncated_normal(
                        bin_val_up[idx],
                        stdp_up[idx] * std_fac,
                        low=bin_val_up[idx],
                        upp=(bin_val_up[idx] + stdp_up[idx] * std_fac))

                    normal_up[idx] = tr_norm.rvs(1)[0]
                rhop_new[filt_vec < 0] = normal_up

            if (len(bin_val[filt_vec > 0]) > 0):

                bin_val_dw = bin_val[filt_vec > 0]
                stdp_dw = bin_std[filt_vec > 0]

                upscale = exag_fac[0]
                std_fac = upscale
                fvec3 = bin_val_dw - stdp_dw * upscale < bounds[0]
                stdp_dw[fvec3] = (
                    (bin_val_dw[fvec3] - bounds[0]) * 0.1)

                normal_dw = np.zeros(len(bin_val_dw))

                for idx in range(0, len(bin_val_dw)):

                    tr_norm = truncated_normal(
                        bin_val_dw[idx],
                        stdp_dw[idx] * std_fac,
                        low=(bin_val_dw[idx] - stdp_dw[idx] * std_fac),
                        upp=bin_val_dw[idx])

                    normal_dw[idx] = tr_norm.rvs(1)[0]
                rhop_new[filt_vec > 0] = normal_dw

            return rhop_new


def computeAverageModel(extracted_data, layer_edges,
                        path_fig=None,
                        n0_offset=0.0,
                        total_offset=0,
                        **kwargs):
    """_summary_

    Args:
        extracted_data (list): Contains the following:
                               Index 0: xmid -> X position along the profile
                               where data has been extracted
                               Index 1: cellcenters_extracted -> x, z pos of
                               the extracted nodes inside the box
                               Index 2: data_extracted -> data corresponding to
                               nodes
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        path_fig (str, optional): Path to folder where figure should be saved.
                                  Defaults to None.
        n0_offset (int, optional): Absolute offset to shift first bin values.
                                   Defaults to 0.
        total_offset (int, optional): Absolute offset to shift all bin values.
                                      Defaults to 0.

    Returns:
        bin_stat, bin_std (np.array, np.array): Computed bin statistic (=mean)
                                                and bin standard deviation
    """

    # below surface
    _, extr_cellcenters, extr_data = extracted_data
    ex_depth = np.max(extr_cellcenters[:, 1]) - extr_cellcenters[:, 1]
    
    # shift the complete data for an offset
    data = extr_data + total_offset

    # bin the data = compute the average model
    bin_stat, _, _ = stats.binned_statistic(ex_depth, data,
                                            statistic=np.mean,
                                            bins=layer_edges
                                            )
    bin_std, _, _ = stats.binned_statistic(ex_depth, data,
                                           statistic=np.std,
                                           bins=layer_edges
                                           )
    # if specified apply a total shift to first depth layer
    # can be used to simulate resistive top layer
    bin_stat[0] = bin_stat[0] + n0_offset

    if path_fig is not None:
        plotAverage1DModelAndScatter(path_fig=path_fig,
                                     extr_data=data,
                                     extr_depths=ex_depth,
                                     layer_edges=layer_edges,
                                     bin_vals=[bin_stat, bin_std],
                                     **kwargs)

    return bin_stat, bin_std


def write2logfile(settings_dict,
                  path_temp,
                  profile,
                  extraction_width,
                  extracted_data):
    """Writes the used settings from the config.json and the location along
    with the used extraction with to a log file located in the same folder as
    the generated 1D models.

    Args:
        settings_dict (dictionary): Settings used for 1D pseudo-random
                                    generation.
        path_temp (string): Path to log file location and models.
        profile (string): File ID of the ERT profile currently being processed.
        extraction_width (float): Used extraction width for the sampling of the
                                  ERT section (i.e., bounding box).
        extracted_data (list): Contains the extracted data at each location,
                               the corresponding cellcenters at each location
                               and the horizontal midpoint of the bounding box.
    """

    logdict = None
    logdict = {"Generator_settings": ''}
    logdict.update({"PATH_RESULTS": path_temp})
    logdict.update({"PATH_SENSORS": settings_dict["PATH_DATA"]
                    + profile + '_grid.csv'})
    logdict.update({"PATH_ERTDATA": settings_dict["PATH_DATA"]
                    + profile + os.sep})
    
    logdict.update({"Extraction_width": extraction_width})
    logdict.update({"ExtractionPoints_X":
                   [extracted_data[idx][0]
                    for idx in range(len(extracted_data))]})

    for option in settings_dict:
        if option in ["PATH_DATA", "PATH_OUT", "FID_LIST",
                      "ER_PLOT_RANGE"]:
            continue
        else:
            logdict.update({option: settings_dict[option]})
  
    with open(path_temp + profile + '_settings.json', 'w') as f:
        f.write(json.dumps(logdict, indent=4))
