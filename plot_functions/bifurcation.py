import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from utopya import DataManager
from utopya.plotting import is_plot_func, PlotHelper, MultiversePlotCreator

from .tools import convert_to_label, find_extremes, setup_figure

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=MultiversePlotCreator)
def bifurcation(dm: DataManager,
                *,
                hlpr: PlotHelper,
                mv_data,
                dim: str,
                plot_kwargs: dict=None,
                title: str=None):

    """For multiverse runs, this plot finds the extremes of the average opinion
    and plots them against the sweep parameter. Only works for a single sweep
    parameter or for a sweep parameter and different seeds.

    Configuration:
        - use the `select/field` key to associate one or multiple datasets
        - choose the dimension `dim` in which the sweep was performed
        - use the `select/subspace` key to set values for all other parameters

    Arguments:
        dm (DataManager): the data manager from which to retrieve the data
        hlpr (PlotHelper): description
        mv_data (xr.Dataset): the extracted multidimensional dataset
        dim (str): the parameter dimension of the diagram
        plot_kwargs (dict, optional): passed to the plot function
        title (str, optional): custom plot title

    Raises:
        TypeError: for a parameter dimension higher than 3 (or 4 if the sweep
        is also conducted over the seed)
        ValueError: if dim does not exist
    """
    if not dim in mv_data.dims:
        raise ValueError(f"Dimension {dim} not available in multiverse data."
                         f" Available: {mv_data.coords}")

    #get datasets and cfg ......................................................
    dataset = mv_data['opinion']
    cfg = dm['multiverse'].pspace.default
    keys = dict(zip(dict(mv_data.dims).keys(), [0]*len(mv_data.dims)))

    for key in ['vertex', 'time', dim]:
        keys.pop(key)

    #assert correct parameterspace dimensionality;
    #manually set any subspace selection parameters in the cfg;
    #for subspace selection: set value to 0 for access during data analysis
    for key in keys:
        if mv_data.dims[key]>1:
            raise ValueError(f"Too many dimensions! Use 'subspace' to "
                             "select specific values for keys other than "
                             f"{dim} and 'seed'!")
        cfg['OpDisc'][key] = mv_data.coords[key].data[0]
        keys[key] = 0

    #figure setup ..............................................................
    figure, axs = setup_figure(cfg, plot_name='bifurcation', title=title, dim=dim)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #data analysis .............................................................
    to_plot = []
    if 'seed' in mv_data.coords and len(dataset['seed'])>1:
        for i in range(len(dataset[dim])):
            extremes = {'const': [], 'osc': []}
            for j in range(len(dataset['seed'])):
                keys[dim] = i
                keys['seed'] = j
                data = np.asarray(dataset[keys])
                means_glob = pd.Series(np.mean(data, axis=1)).rolling(window=5).mean()
                res = find_extremes(means_glob)
                for key in res.keys():
                    extremes[key].append(res[key])
            for key in extremes.keys():
                #flatten
                extremes[key] = [item for sublist in extremes[key] for item in sublist]
            to_plot.append((dataset[dim][i].data, extremes))
    else:
        for i in range(len(dataset[dim])):
            keys[dim] = i
            data = np.asarray(dataset[keys])
            means_glob = np.mean(data, axis=1)
            extremes = find_extremes(means_glob)
            to_plot.append((dataset[dim][i].data, extremes))

    #plotting ..................................................................
    for p, o in to_plot:
        hlpr.ax.scatter([p] * len(o['const']['y']), o['const']['y'],
                        color='peru', s=5, alpha=.05)
        hlpr.ax.scatter([p] * len(o['osc']['y']), o['osc']['y'], color='navy',
                        s=5, alpha=0.05)
    hlpr.ax.set_ylim(0, 1)
    hlpr.ax.set_xlabel(convert_to_label(dim))
    hlpr.ax.set_ylabel(r'mean opinion $\bar{\sigma}$')
    legend_elements = [Line2D([0], [0], label='constant attractors', lw=0,
                       marker='o', color='peru', markerfacecolor='peru',
                       markersize=5),
                      Line2D([0], [0], label='oscillating attractors', lw=0,
                       marker='o', color='navy', markerfacecolor='navy',
                       markersize=5)]
    hlpr.ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.01), loc='lower right',
                   ncol=2, fontsize='xx-small')
