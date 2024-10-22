# -*- coding: utf-8 -*-
import glob
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotPredictedECMaps(df: pd.DataFrame, depth_labels: list,
                        climits: list,
                        path_out: str):
    """Plots maps of the predicted EC distributions and saves the figures.

    Args:
        df (pd.DataFrame): Contains the coordinates and EC values.
        depth_labels (list): Labels for the colorbar.
        climits (list): Limits of the EC values.
        path_out (str): File path where figures should be saved.
    """
    cond_labels = depth_labels
    cond_id = ['D{}'.format(idx) for idx in range(len(cond_labels))]

    for idx, cond in enumerate(cond_labels):
        plt.close('all')
        print('Plotting: {}'.format(cond_id[idx]))
        fig, ax = plt.subplots(figsize=(7.4, 4))

        eca_map = ax.scatter(df['Longitude'],
                             df['Latitude'],
                             s=0.7,
                             c=df[cond_id[idx]],
                             vmin=climits[0],
                             vmax=climits[1],
                             cmap='viridis')
        cb = fig.colorbar(eca_map,
                          orientation='vertical')
        cb.set_label(cond,
                     fontsize=10)
        cb.ax.tick_params(labelsize=10)

        cb.update_ticks()
        xlim = ax.get_xlim()
        ax.set_xticks(np.arange(xlim[0], xlim[1], 200))
        ax.set_xlabel('Longitude [°]', fontsize=10)
        ax.set_ylabel("Latitude [°]", fontsize=10)
        ax.tick_params(labelsize=10)

        ax_hist = fig.add_axes([0.16, 0.17, 0.2, 0.2])

        hist_data = np.log10(df[cond_id[idx]])
        hist_data_clip = np.clip(hist_data, a_min=0, a_max=2)
        ax_hist.hist(hist_data_clip,
                     density=True,
                     bins=50,
                     range=[0, 2],
                     color='k')
        ax_hist.tick_params(left=False, labelleft=False)
        ax_hist.set_xticks([0, 1, 2])
        ax_hist.set_xticklabels(['0', '1', '2'])

        fig.savefig(path_out + "_" + cond_id[idx] + ".png",
                    dpi=300,
                    bbox_inches='tight')


def getDepthLabels(depth: np.array):
    """For a given depth array compute labels with indication of the layer 
    boundaries.

    Args:
        depth (np.array): Contains the depth layer edges.

    Returns:
        depth_labels (list): Contains the labels.
    """
    depth_labels = []
    depthf = depth.flatten()
    for idx in range(len(depthf)-1):
        depth_labels.append(
            'D{}: {:.2f}-{:.2f}'.format(idx, depthf[idx], depthf[idx+1]))
    return depth_labels


def getSubplotDimension(nlayers: int):
    """Based on the number of layers in the 1D EC model define suitable number 
    of rows and columns for the regression subplot.

    Args:
        nlayers (int): Number of depth layers in the 1D EC model.

    Returns:
        nrows, ncols (int, int): Number of rows and columns for the subplot.
    """
    if nlayers == 12:
        nrows, ncols = 3, 4
    elif nlayers == 6:
        nrows, ncols = 3, 2
    elif nlayers == 4:
        nrows, ncols = 2, 2
    else:
        nrows = nlayers
        ncols = 1
    return nrows, ncols


def plotRegressionsTrainingAndPredictedEC(EC_prediction: np.array,
                                          EC_train: np.array,
                                          depth: np.array,
                                          limits: list,
                                          path_plot: str):
    """For each depth plot a regression of the predicted and input/training EC
    values. Also plots the 1:1 line.

    Args:
        EC_prediction (np.array): Contains the predicted 1D EC models.
        EC_train (np.array): Contains the training EC models.
        depth (np.array): Contains the depth layer edges.
        limits (list): EC in S/m limits for all plots.
        path_plot (str): Path to figure. Needs figure name and possibly figure
                         extension (e.g., .png).
    """

    lenEC = np.min(np.shape(EC_prediction))
    nrows, ncols = getSubplotDimension(lenEC)
    depth_label = getDepthLabels(depth)
    limits = [limits[0]/1000, limits[1]/1000]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axesf = axes.flatten()

    # fill the subplots
    for idx in range(lenEC):
        ax = axesf[idx]
        ax.plot(EC_prediction[:, idx], EC_train[:, idx], 'o',
                markersize=1,
                alpha=0.2)
        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), '-k')
        ax.set_title(depth_label[idx], fontsize=10)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.grid(True, linestyle='dotted')

    if lenEC == 12:
        axes[2, 2].set_xlabel('Predicted EC [S/m]')
        axes[1, 0].set_ylabel('Training EC [S/m]')
    elif lenEC == 6:
        axes[2, 1].set_xlabel('Predicted EC [S/m]')
        axes[1, 0].set_ylabel('Training EC [S/m]')
    elif lenEC == 4:
        axes[1, 1].set_xlabel('Predicted EC [S/m]')
        axes[1, 0].set_ylabel('Training EC [S/m]')
    else:
        axesf[-1].set_xlabel('Predicted EC [S/m]')
        axesf[0].set_ylabel('Training EC [S/m]')

    plt.tight_layout()
    fig.savefig(path_plot,
                dpi=300,
                bbox_inches='tight')
    plt.close(fig)


def plotTrainingEvolution(history: pd.DataFrame,
                          evaluation: list,
                          path_figure: str,
                          metrics_and_labels: list):
    """Plot the progress of the passed metrics during the training of the DLN.
    Also plots the corresponding testing set metric in plot title.

    Args:
        history (pd.DataFrame): Contains the epochs and the evolution of the 
                                loss and other metrics of the training and 
                                validation data set.
        evaluation (list): Contains the final loss and other metrics for the
                           testing data set.
        path_figure (str): Path to figure.
        metrics_and_labels (list): Contains the name of label of the metrics in 
                                   the history pd.DataFrame. E.g.
                                   For example: [['mean_squared_error', 'MSE']] 
    """
    n = len(metrics_and_labels)
    if n > 1:
        fig, ax = plt.subplots(n, 1)
        for ii in range(0, n):
            str_tr = metrics_and_labels[ii][0]
            str_val = 'val_' + metrics_and_labels[ii][0]
            str_y = metrics_and_labels[ii][1]
            ax[ii].plot(
                history['epoch'], history[str_tr], '-b',
                label='Training Error: {:.7f}'.format(
                    history[str_tr].iloc[-1]),
                linewidth=2)
            ax[ii].plot(
                history['epoch'], history[str_val], '-r',
                label='Validation Error: {:.7f}'.format(
                    history[str_val].iloc[-1]),
                linewidth=2)
            ax[ii].grid()
            ax[ii].set_xlabel('Epoch')
            ax[ii].set_ylabel(str_y)
            ax[ii].set_title('Testing set ' + str_y +
                             ':{:.7f}'.format(evaluation[ii+1]))
            ax[ii].legend()
    else:
        fig, ax = plt.subplots()
        str_tr = metrics_and_labels[0][0]
        str_val = 'val_' + metrics_and_labels[0][0]
        str_y = metrics_and_labels[0][1]
        ax.plot(
            history['epoch'], history[str_tr], '-b',
            label='Training Error: {:.7f}'.format(
                history[str_tr].iloc[-1]),
            linewidth=2)
        ax.plot(
            history['epoch'], history[str_val], '-r',
            label='Validation Error: {:.7f}'.format(
                history[str_val].iloc[-1]),
            linewidth=2)
        ax.grid()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(str_y)
        ax.set_title('Testing set ' + str_y + ':{:.7f}'.format(evaluation[1]))
        ax.legend()

    plt.tight_layout()
    fig.savefig(path_figure + os.sep + "01_TrainingEvolution.png",
                dpi=300,
                bbox_inches='tight')


def plotAll1DModels(EC: np.array, depth: np.array,
                    clabel: str,
                    fig_title: str,
                    climits: list,
                    path_plot: str):
    """Plot all 1D EC models for a given array next to each other as color
    coded patches.

    Args:
        EC (np.array): Contains the 1D EC models.
        depth (np.array): Contains the depth layers.
        clabel (str): Colorbar label.
        fig_title (str): Plot title.
        climits (list): Colorbar limits in mS/m.
        path_plot (str): Path to figure. Needs figure name and possibly figure
                         extension (e.g., .png).
    """
    fig, ax = plt.subplots(figsize=(13, 3.5))
    EC = EC.T
    x1 = np.arange(0, np.shape(EC)[1]).reshape(1, -1)
    y1 = depth.flatten()[:-1]
    z1 = EC.copy()

    # scale to mS/m
    cf = ax.pcolor(x1, y1, z1 * 1e3,
                   shading='auto',
                   cmap='viridis')
    cf.set_clim(climits)
    fig.colorbar(cf, ax=ax, label=clabel)
    ax.set_xlabel('Model number []')
    ax.set_title(fig_title)
    ax.invert_yaxis()
    ax.set_ylabel('Depth [m]')
    plt.tight_layout()
    fig.savefig(path_plot, dpi=300,
                bbox_inches='tight')
    plt.close(fig)


def plotModelResIPy(ax, mesh, value_id: str, limits: list,
                    sensors=None,
                    **kwargs):
    """Plots the inversion results from ResIPy mesh. Optionally the electrodes
    can be plotted if passed.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        mesh (resipy.mesh): ResIPy mesh class holding the x y z position of the
                            mesh nodes and node data.
        value_id (str): Mesh property to plot.
        limits (list): Colorbar limits for mesh property to plot.
        sensors (np.array, optional): Contains x, z position of electrodes.
                            Defaults to None.
        **kwargs: matplotlib.pyplot kwargs can be passed.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
    """

    mesh.show(attr=value_id,
              color_map='viridis',
              ax=ax,
              edge_color=None,
              vmin=limits[0],
              vmax=limits[1],
              clabel=kwargs.get('cbar_label', r'|$\rho$| [$\Omega$m]'))

    ax.set_xlabel(kwargs.get('xlabel', 'x [m]'))
    ax.set_ylabel(kwargs.get('ylabel', 'z [m]'))
    ax.grid(True, linestyle='dotted')

    if sensors is not None:
        ax = plotElectrodes(ax, sensors, marker='o',
                            color='k',
                            ms=kwargs.get('markersize', 1.5),
                            linewidth=kwargs.get('linewidth', 0.2),
                            **kwargs)

    return ax


def plotElectrodes(ax, electrodes, **kwargs):
    """Plot the passed array containing x, z positions of the electrodes to
    matplotlib.axes.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        electrodes (np.array): Contains x, z position of electrodes.
        **kwargs: matplotlib.pyplot kwargs can be passed.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
    """
    ax.plot(electrodes[:, 0],
            electrodes[:, 1],
            **kwargs)

    return ax


def indicateExtractionAreas(ax, area_id, extracted_data, **kwargs):
    """Computes a bounding rectangle around passed mesh nodes/cellcenters data
    and draws the rectangle to the passed matplotlib.axes. Also plots the
    positions of the nodes/cellcenters within the bounding rectangle.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        area_id (int): Number of extraction area. Used as label over box.
        extracted_data (list): Contains the following:
                               Index 0: xmid -> X position along the profile
                               where data has been extracted
                               Index 1: cellcenters_extracted -> x, z pos of
                               the extracted nodes inside the box
                               Index 2: data_extracted -> data corresponding to
                               nodes
        **kwargs: matplotlib.pyplot kwargs can be passed.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
    """

    xmid, cellcenters, _ = extracted_data
    bottom_left = [np.min(cellcenters[:, 0]), np.min(cellcenters[:, 1])]
    width = np.max(cellcenters[:, 0]) - np.min(cellcenters[:, 0])
    height = np.max(cellcenters[:, 1]) - np.min(cellcenters[:, 1])
    ax.plot(cellcenters[:, 0], cellcenters[:, 1], 'o',
            markersize=kwargs.get('markersize', 0.1),
            **kwargs)

    rect = Rectangle(bottom_left, width, height,
                     linewidth=kwargs.get('linewidth', 1),
                     edgecolor=kwargs.get('edgecolor', 'r'),
                     facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.annotate(str(area_id+1), [xmid, np.max(cellcenters[:, 1]) + 1])
    ylimc = ax.get_ylim()
    ax.set_ylim(ylimc[0], ylimc[1]+1.0)

    return ax


def prepare1DStepmodel(layer_edges, bin_values):
    """Prepare data to plot 1D stepmodels from layer edges and associated
    layer values.

    Args:
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        bin_values (np.array): Array containing the value of each depth layer.
                               Size n-1x1

    Returns:
        profile (np.array): Array containing the postive depths of the step
                            model in pairs: [d1, d1, d2, d2, ... , dn, dn].
                            Size n*2x1
        rhop (np.array): Contains the associated values of the model.
                         Size n*2x1
    """

    profile = np.zeros((len(layer_edges) * 2))
    profile[0::2] = layer_edges
    profile[1:-1:2] = layer_edges[1:]
    profile = profile[:-2]

    rhop = np.zeros((len(bin_values) * 2))
    rhop[0::2] = bin_values
    rhop[1::2] = bin_values

    return profile, rhop


def plotStepModel(ax, step_depths, step_values, **kwargs):
    """Plots the 1D step model with negative depths.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        step_depths (np.array): Array containing the postive depths of the
                                step model in pairs:
                                [d1, d1, d2, d2, ... , dn, dn]
        step_values (np.array): Contains the associated values of the model.
        **kwargs: matplotlib.pyplot kwargs can be passed.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
    """

    # use negative depths!
    ax.set_ylim([-np.max(step_depths), 0])
    ax.plot(step_values,
            -step_depths,
            color=kwargs.pop('color', 'k'),
            linewidth=kwargs.pop('linewidth', 1.0),
            linestyle=kwargs.pop('linestyle', '-'),
            **kwargs)
    ax.grid(True, linestyle='dotted')

    return ax


def plotScatterAndStandardDeviations(ax,
                                     extr_data, extr_depths,
                                     layer_edges, bin_vals,
                                     cmap):
    """Scatterplots the raw extracted values as a function of depth and the
    corresponding 1D step model after binning (= average model). Moreover,
    a boxes defining two times the negative and positive standard deviation
    are plotted as grey areas.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        extr_data (np.array): Array containing the extracted inversion
                                results for the specified bounding box.
        extr_depths (np.array): Array containing the depth values associated
                                to the extracted inversion results for the
                                specified bounding box.
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        bin_vals (list): Binning results using layer edges and extra_data.
                            Contains a list of two np.arrays containing the bin
                            statistic (default "mean") and bin
                            standard deviation.
        cmap (str): Matplotlib colormap used for scatter.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
    """
    bin_stat, bin_std = bin_vals
    # use negative depths
    ax.scatter(extr_data, -extr_depths, c=extr_data, cmap=cmap)
    # plot the depth layer edges
    # beware magic numbers!
    xarray = np.array(np.ones(len(layer_edges)) * np.min(extr_data))-5
    ax.plot(xarray, np.array(layer_edges)*-1, marker=5, c='k')

    profile, rhop = prepare1DStepmodel(layer_edges, bin_stat)
    _, stdp = prepare1DStepmodel(layer_edges, bin_std)
    stdp_plus = rhop + 2 * stdp  # use 2x standard deviation for boxes
    stdp_minus = rhop - 2 * stdp

    # plot the range between -2x standard deviatio - 2x standard deviation
    ax.fill_betweenx(-profile, stdp_minus, stdp_plus,
                     facecolor='Grey', interpolate=True,
                     alpha=0.1,
                     zorder=1)
    ax.plot(stdp_plus, -profile, '-', linewidth=0.3, c='Grey')
    ax.plot(stdp_minus, -profile, '-', linewidth=0.3, c='Grey')

    # plot the average model = binning result
    ax.plot(rhop, -profile, '--k')
    ax.grid(True, linestyle='dotted')

    return ax


def plotModelPatchCollectionAndColorbar(ax, layer_edges, bin_stat,
                                        cmap, limits):
    """Plots the 1D model as a collection of patches and adds a colorbar next
    to it.

    Args:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        layer_edges (np.array): Array containing the upper and lower depth
                                    values of each depth layer. Size nx1
        bin_stat (np.array): Array that contains the bin values = layer values.
        cmap (str): Matplotlib colormap used for the patches and colorbar.
        limits (list): Value limits for the PatchCollection and colorbar.

    Returns:
        ax (matplotlib.axes): (Sub)plot of figure to plot the data.
        cb (Figure.colorbar): Matplotlib colorbar object.
    """
    pr = getModelPatchCollection(layer_edges)
    pr.set_array(bin_stat)
    pr.set_cmap(cmap=cmap)
    ax.add_collection(pr)
    pr.set_clim(limits)
    ax.plot()
    ax.set_xlim([0, 1])
    ax.tick_params(labelleft=False, labelbottom=False)

    # add extra axes for colorbar, needed to match the cb size to plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='35%', pad=0.1)
    fig = ax.get_figure()
    cb = fig.colorbar(pr,
                      ax=ax,
                      cax=cax,
                      orientation='vertical')

    return ax, cb


def plotAverage1DModelAndScatter(path_fig, extr_data, extr_depths,
                                 layer_edges, bin_vals, **kwargs):
    """Scatterplots the raw extracted values as a function of depth and the
    corresponding 1D step model after binning (= average model). Also plots the
    average model as a color coded collection of individual patches. The depth
    layer edges are indicated by black triangles to the left of the plot.

    Args:
        path_fig (str): Path to folder where plot should be saved.
        extr_data (np.array): Array containing the extracted inversion results
                              for the specified bounding box.
        extr_depths (np.array): Array containing the depth values associated to
                                the extracted inversion results for the
                                specified bounding box.
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        bin_vals (list): Binning results using layer edges and extra_data.
                         Contains a list of two np.arrays containing the bin
                         statistic (default "mean") and bin standard deviation.

        ax1_xlabel (str, optional): X label for left subplot.
                                   Defaults to r"$\rho$ [$\Omega$m]"
        ax2_xlabel (str, optional): X label for right subplot.
                                   Defaults to "Model\nvalues".
        ylabel (str, optional): Y label for both subplots.
        cmap (str, optional): Matplotlib colormap used for scatter and patches.
        **kwargs: matplotlib.pyplot kwargs can be passed.
    """

    depth = np.max(np.array(layer_edges))

    # setup figure for plotting
    fig = plt.figure(figsize=(5.6, 5.5))
    gs = gridspec.GridSpec(2, 3)  # 7
    gs.update(wspace=0.2)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)
    # get **kwargs if provided
    ax1_label = kwargs.get('ax1_label', r"$\rho$ [$\Omega$m]")
    ax2_label = kwargs.get('ax2_label', 'Model\nvalues')
    y_label = kwargs.get('y_label', 'Depth [m]')
    cmap = kwargs.get('cmap', 'viridis')

    ax1 = plotScatterAndStandardDeviations(ax1, extr_data, extr_depths,
                                           layer_edges, bin_vals, cmap)

    # set limits and labels for ax1
    ax1.set_ylim([-depth, 0])
    ax1.set_xlabel(ax1_label)
    ax1.set_ylabel(y_label)

    limits = [np.min(extr_data), np.max(extr_data)]
    ax2, cb = plotModelPatchCollectionAndColorbar(ax2, layer_edges,
                                                  bin_vals[0],
                                                  cmap,
                                                  limits)
    # set label of colorbar
    cb.set_label(ax1_label)
    ax2.set_xlabel(ax2_label)

    # finally save the figure
    if os.path.exists(path_fig + 'extraction01.png'):
        # now check the current number
        files_there = glob.glob(path_fig + 'extraction*.png')
        max_num = max([int(file.split(os.sep)[-1].split('.')[-2][-2:])
                       for file in files_there])
        fig.savefig(path_fig + 'extraction'
                    + str(max_num + 1).zfill(2) + '.png',
                    bbox_inches='tight')
    else:
        fig.savefig(path_fig + 'extraction01.png',
                    bbox_inches='tight')

    plt.close(fig)


def getModelPatchCollection(layer_edges, xvalues=[0, 1]):
    """Compute for a given array of depth layers a collection of patches that
    can be used to plot a color coded depth model (in comparison to step
    model).

    Args:
        layer_edges (np.array): Array containing the upper and lower depth
                                values of each depth layer. Size nx1
        xvalues (list, optional): Horizontal range of patches.
                                  Defaults to [0, 1].

    Returns:
        p (matplotlib.collections.PatchCollection): PatchCollection that
                                                    contains the individual
                                                    patches for each depth.
    """
    patches = []
    x1, x2 = xvalues
    layer_edges_invert = np.array(layer_edges) * -1
    # print(layer_edges_invert)
    # loop to compute polygon patches
    for idx in range(0, len(layer_edges) - 1):
        # print(idx)
        y1, y2 = layer_edges_invert[idx], layer_edges_invert[idx + 1]

        poly = Polygon([
            [x1, y1],  # origin
            [x2, y1],  # to bottom right
            [x2, y2],  # to top right
            [x1, y2],  # to top left
            [x1, y1]])  # to origin

        patches.append(poly)

    p = PatchCollection(patches,
                        edgecolors='none',
                        linewidth=0)
    return p
