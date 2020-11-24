import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from utopya import DataManager
from utopya.plotting import is_plot_func, PlotHelper, MultiversePlotCreator

from .data_analysis import find_extrema
from .tools import convert_to_label, get_keys_cfg, setup_figure

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=MultiversePlotCreator)
def bifurcation(dm: DataManager,
                *,
                hlpr: PlotHelper,
                mv_data,
                avg_window: int=20,
                dim: str=None,
                plot_kwargs: dict=None,
                title: str=None):

    """For multiverse runs, this plot finds the upper turning points of the
    average opinion and plots them against the sweep parameter. Only works for a
    single sweep parameter or for a sweep parameter and different seeds.

    Configuration:
        - use the `select/field` key to associate one or multiple datasets
        - choose the dimension `dim` in which the sweep was performed. For a single
          sweep dimension, the sweep parameter is automatically deduced
        - use the `select/subspace` key to set values for all other parameters

    Arguments:
        dm (DataManager): the data manager from which to retrieve the data
        hlpr (PlotHelper): description
        mv_data (xr.Dataset): the extracted multidimensional dataset
        avg_window (int): the smoothing window for the rolling average of the
           opinion dataset
        dim (str, optional): the parameter dimension of the diagram. If no str
           is passed, an attempt will be made to automatically deduce the sweep
           dimension.
        plot_kwargs (dict, optional): kwargs passed to the scatter plot function
        title (str, optional): custom plot title

    Raises:
        ValueError: for a parameter dimension higher than 3 (or 4 if the sweep
        is also conducted over the seed)
        ValueError: if dim does not exist
    """
    if dim is None:
        dim = deduce_sweep_dimension(mv_data)

    else:
        if not dim in mv_data.dims:
            raise ValueError(f"Dimension '{dim}' not available in multiverse data."
                             f" Available: {mv_data.coords}")

    #get datasets and cfg ......................................................
    dataset = mv_data['opinion']
    time_steps = dataset['time'].size
    keys, cfg = get_keys_cfg(mv_data, dm['multiverse'].pspace.default,
                             keys_to_ignore=[dim, 'time'])

    #figure setup ..............................................................
    figure, axs = setup_figure(cfg, plot_name='bifurcation', title=title, dim1=dim)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #data analysis .............................................................
    #get the turning points of the average opinion (maxima only). If a sweep over
    #seed was performed, multiple datapoints are collected per x-value
    log.info("Starting data analysis ...")
    to_plot = []
    if 'seed' in mv_data.coords and len(dataset['seed'])>1:
        for i in range(len(dataset[dim])):
            extremes = []
            for j in range(len(dataset['seed'])):
                keys[dim] = i
                keys['seed'] = j
                data = np.asarray(dataset[keys])
                means_glob = pd.Series(np.mean(data, axis=1)).rolling(window=avg_window).mean()
                res = find_extrema(means_glob)
                extremes.extend(res['max']['y'])

            to_plot.append((dataset[dim][i].data, extremes))
    else:
        for i in range(len(dataset[dim])):
            keys[dim] = i
            data = np.asarray(dataset[keys])
            means_glob = pd.Series(np.mean(data, axis=1)).rolling(window=avg_window).mean()
            extremes = find_extrema(means_glob)['max']['y']

            to_plot.append((dataset[dim][i].data, extremes))

    log.info("Data analysis complete.")

    #plot scatter plot of extrema ..............................................
    for p, o in to_plot:
        hlpr.ax.scatter([p] * len(o), o, **plot_kwargs)

    #hlpr.ax.set_ylim(0, 1)
    hlpr.ax.set_xlabel(convert_to_label(dim))
    hlpr.ax.set_ylabel(r'mean opinion $\bar{\sigma}$')
    legend_elements = [Line2D([0], [0],
                       label=(r'$\bar{\sigma}^\prime = 0$,'+
                              r'$\bar{\sigma}^{\prime \prime} < 0$'),
                       lw=0,
                       marker='o', color=plot_kwargs['color'] if not None else 'navy',
                       markerfacecolor=plot_kwargs['color'] if not None else 'navy',
                       markersize=5)]
    hlpr.ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.01),
                   loc='lower right', ncol=2, fontsize='xx-small')
