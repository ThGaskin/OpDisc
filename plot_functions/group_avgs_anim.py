import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utopya import DataManager
from utopya.plotting import MultiversePlotCreator, PlotHelper, is_plot_func

from .data_analysis import data_by_group
from .tools import convert_to_label, deduce_sweep_dimension, get_keys_cfg, R_p, setup_figure

log = logging.getLogger(__name__)
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

# ------------------------------------------------------------------------------
@is_plot_func(creator_type=MultiversePlotCreator, supports_animation=True,
              helper_defaults=dict(set_labels=dict(x="User opinion", y="Time")))
def group_avgs_anim(dm: DataManager, *,
                   hlpr: PlotHelper,
                   mv_data,
                   dim: str=None,
                   age_groups: list=[10, 20, 40, 60, 80],
                   num_bins: int=100,
                   title: str=None,
                   val_range: tuple=(0, 1),
                   write: bool=False):
    """Plots an animation of the average opinion evolution by group as a
    function of the sweep parameter.

    Arguments:
        dm (DataManager): the data manager from which to retrieve the data
        hlpr (PlotHelper): description
        mv_data (xr.Dataset): the extracted multidimensional dataset
        dim (str, optional): the parameter dimension of the diagram. If no str
           is passed, an attempt will be made to automatically deduce the sweep
           dimension.
        num_bins (int, optional): binning size for the histogram
        title (str, optional): custom plot title
        val_range (tuple, optional): binning range for the histogram
        write (bool, optional): if true, the model will write the widths of the
           distribution at each time step and with corresponding R_p factors
           (for the purposes of my thesis only)
    Raises:
        ValueError: if the dimension is not present in the multiverse data
        ValueError: if the parameter space is greater than four
        ValueError: if the sweep parameter is 'seed' (to do)
    """
    if dim is None:
        dim = deduce_sweep_dimension(mv_data, key_to_ignore='')
    else:
        if not dim in mv_data.dims:
            raise ValueError(f"Dimension '{dim}' not available in multiverse data."
                             f" Available: {mv_data.coords}")
    if len(mv_data.dims)>3:
        for key in mv_data.dims.keys():
            if key not in ['vertex', 'time', dim] and mv_data.dims[key]>1:
                raise ValueError(f"Too many dimensions! Use 'subspace' to "
                           f"select specific values for keys other than {dim}!")

    if dim=='seed':
        raise ValueError("'seed' sweeps currently not supported.")


    #datasets...................................................................
    keys, cfg = get_keys_cfg(mv_data, dm['multiverse'].pspace.default,
                             keys_to_ignore=[dim, 'time'])
    mode = cfg['OpDisc']['mode']
    ageing = True if mode=='ageing' else False
    #get group labels
    if ageing:
        #group labels change over time
        keys.update({dim: 0})
        groups = np.asarray(mv_data['group_label'][keys], dtype=int)
        keys.pop(dim)
    else:
        #group labels do not change over time
        keys.update({'time':0, dim: 0})
        groups = np.asarray(mv_data['group_label'][keys], dtype=int)
        [keys.pop(ele) for ele in ['time', dim]]
    num_groups = len(age_groups)-1 if ageing else cfg['OpDisc']['number_of_groups']
    num_vertices = cfg['OpDisc']['nw']['num_vertices']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]
    time_steps = mv_data['time'].size
    time = mv_data['time'].data

    #figure layout .............................................................
    figure, axs = setup_figure(cfg, plot_name='group_avgs_anim', title=title, dim1=dim)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #data analysis .............................................................
    #get mean opinion and std of each group using tools.data_by_group
    means = np.zeros((len(mv_data.coords[dim]), time_steps, num_groups))
    stddevs = np.zeros_like(means)
    for param in range(len(mv_data.coords[dim])):
        keys[dim] = param
        data = np.asarray(mv_data[keys]['opinion'])
        data_by_groups = data_by_group(data, groups, group_list, val_range,
                                        num_bins, ageing=ageing)
        for k in range(num_groups):
            for t in range(time_steps):
                means[param, t, k]=np.mean(data_by_groups[k][t])
                stddevs[param, t, k]=np.std(data_by_groups[k][t])
    log.info("Finished data analysis.")

    #plotting...................................................................
    #get pretty labels
    if ageing:
        labels = [f"Ages {group_list[_]}-{group_list[_+1]}" for _ in range(num_groups)]
        max_age = np.amax(groups)
        if (age_groups[-1]>=max_age):
            labels[-1]=f"Ages {group_list[-2]}+"
    else:
        labels = [f"Group {_+1}" for _ in range(num_groups)]

    #calculate R_p factor (for p_hom sweeps)
    if mode not in ['ageing', 'conflict_dir', 'conflict_undir']:
        R_p_fs = R_p(mv_data.coords[dim], num_groups, mode)

    #animate
    def update_data(stepsize: int=1):
        log.info(f"Plotting animation with {len(mv_data.coords[dim])} frames ...")
        for param in range(len(mv_data.coords[dim])):
            hlpr.ax.clear()
            hlpr.ax.set_xlim(0, 1)
            hlpr.ax.set_ylim(time[-1], 0)
            hlpr.ax.set_xlabel(hlpr.axis_cfg['set_labels']['x'])
            hlpr.ax.set_ylabel(hlpr.axis_cfg['set_labels']['y'])
            if dim=='homophily_parameter':
                if mode not in ['ageing', 'conflict_dir', 'conflict_undir']:
                    sw_text = (f"$R_p=${R_p_fs[param]:.3f} ({convert_to_label(dim)} = {mv_data[dim][param].data})")
                else:
                    sw_text = f"{convert_to_label(dim)} = {mv_data[dim][param].data}"
            else:
                sw_text = f"{convert_to_label(dim)}={mv_data[dim][param].data}"
            sweep_text = hlpr.ax.text(0, 1.02, sw_text, fontsize='x-small',
                                                    transform=hlpr.ax.transAxes)
            for i in range(num_groups):
                hlpr.ax.errorbar(means[param, :, i], time,
                                 xerr=stddevs[param, :, i], lw=2,
                                 alpha=1, elinewidth=0.1, label=labels[i],
                                 capsize=1, capthick=0.3)
            hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                           ncol=num_groups, fontsize='xx-small')
            yield
    hlpr.register_animation_update(update_data)

    #write data values for further evaluation....................................
    #This is for the purpose of my thesis only and will be removed upon
    #completion.
    if write and dim=='homophily_parameter':
        widths = np.zeros((time_steps, len(mv_data.coords[dim])))
        w_0 = np.min(means[:, -1, -1]-means[:, -1, 0])
        w_max = np.max(means[-1, :, -1]-means[-1, :, 0])
        for param in range(len(mv_data.coords[dim])):
            widths[:, param] = (means[param, :, -1]-means[param, :, 0]-w_0)/(w_max-w_0)
        df = pd.DataFrame(widths, time, R_p_fs)
        df.to_csv(hlpr.out_path.replace('group_avgs_anim.mp4', f'widths_{mode}.csv'))
        log.info("Finished writing files")
