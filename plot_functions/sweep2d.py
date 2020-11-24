import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utopya import DataManager
from utopya.plotting import is_plot_func, PlotHelper, MultiversePlotCreator

from .data_analysis import get_absolute_area, get_area, avg_of_means_stddevs, difference_of_extreme_means
from .tools import convert_to_label, get_keys_cfg, parameters, R_p, setup_figure

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=MultiversePlotCreator)
def sweep2d(dm: DataManager,
          *,
          hlpr: PlotHelper,
          mv_data,
          age_groups: list=[10, 20, 40, 60, 80],
          x: str,
          y: str,
          plot_kwargs: dict={},
          stacked: bool=False,
          to_plot: str):

    """For multiverse runs, this produces a two dimensional plot showing
    specified values.

    Configuration:
        - use the `select/field` key to associate one or multiple datasets
        - choose the dimension `dim` in which the sweep was performed. For a single
          sweep dimension, the sweep parameter is automatically deduced
        - use the `select/subspace` key to set values for all other parameters

    Arguments:
        dm (DataManager): the data manager from which to retrieve the data
        hlpr (PlotHelper): description
        mv_data (xr.Dataset): the extracted multidimensional dataset
        age_groups (list): The age intervals to be plotted in the final_ax
            distribution plot for the 'ageing' mode.
        x (str): the first parameter dimension of the diagram.
        y (str): the first parameter dimension of the diagram.
        plot_kwargs (dict, optional): kwargs passed to the scatter plot function
        stacked (bool): whether to plot a 2d heatmap or a stacked line plot
        to_plot (str): the data to be plotted. Can be:
            - extreme_means_diff: the difference between the means of the outer
              groups at the final time step. not compatible with a seed sweep.
            - avg_of_means_diff_to_05: the average of the absolute distance of
              each group to 0.5 at the final time step
            - absolute_area: the area (unsigned) of the mean minus 0.5 of the
              entire population. is always positive.
            - area: the area (signed) of the mean minus 0.5 of the entire
              population. can be positive or negative.
            - area_diff: the difference of the absolute area und the signed
              area under the means curve.

    Raises:
        ValueError: if a sweep over 'seed' is performed and to_plot is
        'extreme_means_diff'.
    """
    if ((to_plot == 'extreme_means_diff') and ('seed' in mv_data.coords) and
         (len(mv_data.coords['seed'].data)>1)):
        raise ValueError("Plotting does not support 'seed' at this time. Select"
                         " a single value using the 'subspace' key")

    #get datasets and cfg ......................................................
    keys, cfg = get_keys_cfg(mv_data, dm['multiverse'].pspace.default,
                                                          keys_to_ignore=[x, y])
    mode = cfg['OpDisc']['mode']
    ageing = True if mode == 'ageing' else False
    num_groups = len(age_groups)-1 if ageing else cfg['OpDisc']['number_of_groups']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]

    requires_group_label = ['extreme_means_diff', 'avg_of_means_diff_to_05', 'avg_of_stddevs']
    #get group labels
    if ageing:
        #group labels change over time
        keys.update({x: 0, y: 0})
        if to_plot in requires_group_label:
            groups = np.asarray(mv_data['group_label'][keys], dtype=int)
        for ele in [x, y]:
            keys.pop(ele)
    else:
        #group labels do not change over time
        keys.update({'time':0, x: 0, y: 0})
        if to_plot in requires_group_label:
            groups = np.asarray(mv_data['group_label'][keys], dtype=int)
        for ele in [x, y, 'time']:
            keys.pop(ele)

    #figure setup ..............................................................
    figure, axs = setup_figure(cfg, plot_name=to_plot, dim1=x, dim2=y)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #data analysis .............................................................
    if to_plot == 'extreme_means_diff':
        data_to_plot = difference_of_extreme_means(mv_data, x, y, groups, group_list,
                             ageing=ageing, group_1=0, group_2=-1, time_step=-1)

    elif to_plot == 'avg_of_means_diff_to_05':
        data_to_plot = avg_of_means_stddevs(mv_data, x, y, groups, group_list,
             keys, mode, num_groups, which='means', ageing=ageing, time_step=-1)

    elif to_plot == 'avg_of_stddevs':
        data_to_plot = avg_of_means_stddevs(mv_data, x, y, groups, group_list,
           keys, mode, num_groups, which='stddevs', ageing=ageing, time_step=-1)

    elif to_plot == 'absolute_area':
        data_to_plot = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))
        for param1 in range(len(mv_data.coords[x])):
            keys[x] = param1
            data_to_plot[:, param1] = get_absolute_area(mv_data, keys, dim=y)[0]

    elif to_plot == 'area':
        data_to_plot = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))
        for param1 in range(len(mv_data.coords[x])):
            keys[x] = param1
            data_to_plot[:, param1] = get_area(mv_data, keys, dim=y)[0]

    elif to_plot == 'area_diff':
        data_to_plot = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))
        for param1 in range(len(mv_data.coords[x])):
            keys[x] = param1
            val_1 = get_absolute_area(mv_data, keys, dim=y)[0]
            val_2 = get_area(mv_data, keys, dim=y)[0]
            data_to_plot[:, param1] = np.subtract(val_1, val_2)

    #plotting ..................................................................
    if stacked:
        for i in range(len(mv_data.coords[y])):
            hlpr.ax.plot(data_to_plot[i, :],
                    label=f'{convert_to_label(y)}={mv_data.coords[y].data[i]}',
                    **plot_kwargs)
        hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                       ncol=len(mv_data.coords[y])+1, fontsize='xx-small')

    else:
        df = pd.DataFrame(data_to_plot, index=mv_data.coords[y].data,
                                                     columns=mv_data.coords[x].data)
        im = hlpr.ax.pcolor(df, **plot_kwargs)
        hlpr.ax.set_ylabel(parameters[y], rotation=0)
        hlpr.ax.set_yticks([i for i in np.linspace(0.5,
                     len(mv_data.coords[y].data)-0.5, len(mv_data.coords[y].data))])
        hlpr.ax.set_yticklabels([np.around(i, 3) for i in mv_data.coords[y].data])

        hlpr.ax.set_xlabel(parameters[x])
        hlpr.ax.set_xticks([i for i in np.linspace(0.5,
                     len(mv_data.coords[x].data)-0.5, len(mv_data.coords[x].data))])
        hlpr.ax.set_xticklabels([np.around(i, 3) for i in mv_data.coords[x].data])

        divider = make_axes_locatable(hlpr.ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = figure.colorbar(im, cax=cax)

        cbar.set_label(convert_to_label(to_plot))
