import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utopya import DataManager
from utopya.plotting import is_plot_func, PlotHelper, MultiversePlotCreator

from .data_analysis import get_absolute_area, get_area, means_stddevs_by_group
from .tools import convert_to_label, deduce_sweep_dimension, get_keys_cfg, setup_figure

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=MultiversePlotCreator)
def sweep1d(dm: DataManager,
                *,
                hlpr: PlotHelper,
                mv_data,
                age_groups: list=[10, 20, 40, 60, 80],
                dim: str=None,
                plot_by_groups: bool=True,
                plot_kwargs: dict={},
                to_plot: str):

    """Plots statistical measures of the final distribution over a sweep
    parameter.

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
        dim (str, optional): the parameter dimension of the diagram. If none is
            provided, an attempt will be made to automatically deduce the sweep
            parameter
        plot_kwargs (dict): kwargs passed to the errorbar plot function
        to_plot (str): the data to be plotted. Can be:
            - absolute_area: the area (unsigned) of the mean minus 0.5 of the
              entire population. is always positive.
            - area: the area (signed) of the mean minus 0.5 of the entire
              population. can be positive or negative.
            - means: the mean of each group at the final time step, with an error
            - stddevs: the stddev of each group at the final time step, with an
              error

    Raises:
        ValueError: if an unknown 'to_plot' argument is passed
        ValueError: if the dimension cannot be deduced
        ValueError: if the dimension is not available
    """

    if to_plot not in ['absolute_area', 'area', 'area_comp', 'area_diff', 'means', 'stddevs']:
        raise ValueError(f"Unknown statistical variable {to_plot}!")

    if dim is None:
        dim = deduce_sweep_dimension(mv_data)
    else:
        if not dim in mv_data.dims:
            raise ValueError(f"Dimension '{dim}' not available in multiverse data."
                             f" Available: {mv_data.coords}")


    #get datasets and cfg ......................................................
    keys, cfg = get_keys_cfg(mv_data, dm['multiverse'].pspace.default,
                             keys_to_ignore=[dim, 'time'])
    mode = cfg['OpDisc']['mode']
    ageing = True if mode=='ageing' else False
    num_groups = len(age_groups)-1 if ageing else cfg['OpDisc']['number_of_groups']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]
    #get pretty labels
    if ageing:
        labels = [f"Ages {group_list[_]}-{group_list[_+1]}" for _ in range(num_groups)]
        max_age = np.amax(mv_data[keys]['group_label'])
        if (age_groups[-1]>=max_age):
            labels[-1]=f"Ages {group_list[-2]}+"
    else:
        labels = [f"Group {_+1}" for _ in group_list]

    #figure setup ..............................................................
    figure, axs = setup_figure(cfg, plot_name=to_plot, dim1=dim)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #data analysis and plotting ................................................
    log.info("Commencing data analytics ...")

    if to_plot == 'absolute_area':
        data_to_plot, err = get_absolute_area(mv_data, keys, dim=dim)
        hlpr.ax.errorbar(mv_data.coords[dim].data, data_to_plot, yerr=err, **plot_kwargs)

    elif to_plot == 'area':
        data_to_plot, err = get_area(mv_data, keys, dim=dim)
        hlpr.ax.errorbar(mv_data.coords[dim].data, data_to_plot, yerr=err, **plot_kwargs)

    elif to_plot == 'area_comp':
        data_to_plot_0, err_0 = get_absolute_area(mv_data, keys, dim=dim)
        # hlpr.ax.errorbar(mv_data.coords[dim].data, data_to_plot_0, yerr=err_0, **plot_kwargs, label=r'$\vert A \vert$')
        data_to_plot_1, err_1 = get_area(mv_data, keys, dim=dim)
        # hlpr.ax.errorbar(mv_data.coords[dim].data, data_to_plot_1, yerr=err_1, **plot_kwargs, label=r'$A$')
        # hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
        #                ncol=2, fontsize='xx-small')
        #write data values for further evaluation....................................

        res = np.vstack((data_to_plot_0, err_0, data_to_plot_1, err_1))
        idx = ['abs_area', 'abs_area_err', 'area', 'area_err']
        df = pd.DataFrame(res, idx, mv_data.coords[dim].data)
        phom = cfg['OpDisc']['homophily_parameter']
        df.to_csv(hlpr.out_path.replace('area.pdf', f'area_N_{num_groups}_phom_{phom}.csv'))
        log.info("Finished writing files")

    elif to_plot == 'area_diff':
        data_to_plot_0, err_0 = get_absolute_area(mv_data, keys, dim=dim)
        data_to_plot_1, err_1 = get_area(mv_data, keys, dim=dim)
        hlpr.ax.plot(mv_data.coords[dim].data, np.subtract(data_to_plot_0, data_to_plot_1), **plot_kwargs)


    elif to_plot == 'means' or to_plot=='stddevs':
        keys['time']=-1
        data_to_plot, err = means_stddevs_by_group(mv_data, group_list, dim, keys,
                          mode=mode, ageing=ageing, num_groups=num_groups, which=to_plot, time_step=-1)
        if plot_by_groups:
            for i in range(len(err)):
                hlpr.ax.errorbar(mv_data.coords[dim].data, data_to_plot[i], yerr=err[i],
                                 label=labels[i], **plot_kwargs)
            hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                           ncol=num_groups+1, fontsize='xx-small')
        else:
            hlpr.ax.errorbar(mv_data.coords[dim].data, np.mean(data_to_plot, axis=0),
                            yerr=np.mean(err, axis=0), **plot_kwargs)

    log.info("Data analysis complete.")

    #set axis lables etc .......................................................
    hlpr.ax.set_xlabel(convert_to_label(dim))
    hlpr.ax.set_ylabel(convert_to_label(to_plot))
