import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

from utopya import DataManager, UniverseGroup
from utopya.plotting import is_plot_func, UniversePlotCreator, PlotHelper

from .data_analysis import data_by_group
from .tools import setup_figure

#matplotlib.rcParams['mathtext.fontset']='stix'
#matplotlib.rcParams['font.family']='serif'
#rc('text', usetex=True)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator)
def opinion_at_time(dm: DataManager,
                    *,
                    hlpr: PlotHelper,
                    uni: UniverseGroup,
                    age_groups: list=[10, 20, 40, 60, 80],
                    num_bins: int=100,
                    plot_kwargs: dict={},
                    time_step: float,
                    title: str=None,
                    to_plot: str,
                    val_range: tuple=(0., 1.)):
    """Plots opinion state at a specific time frame. If the model mode is 'ageing',
    the opinions of the specified age groups are plotted, in all other cases
    the opinions of the groups are shown.

    Arguments:
        age_groups (list): The age intervals to be plotted in the axs[2]
            distribution plot for the 'ageing' mode.
        time_step (int): the time frame to plot (as a fraction of the total length)
        title (str, optional): Custom plot title
        to_plot (str): whether or not to differentiate by groups

    Raises:
        TypeError: if the age groups have fewer than two entries (in which
        case no binning is possible)
        ValueError: if to_plot or time_step are invalid
    """
    if len(age_groups)<2:
        raise TypeError("'Age groups' list needs at least two entries!")

    if to_plot not in ['overall', 'by_group', 'discriminators']:
        raise ValueError(f"Unrecognized argument {to_plot}: must be one of "
                          "'overall' or 'by_group'")
    if time_step<0 or time_step>1:
        raise ValueError("time_step must be in [0, 1]")

    #datasets...................................................................
    mode = uni['cfg']['OpDisc']['mode']
    ageing = True if mode=='ageing' else False
    time_idx = int(time_step*(uni['data/OpDisc/nw/opinion']['time'].size-1))
    opinions = uni['data/OpDisc/nw/opinion']
    if ageing:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'], dtype=int)
    else:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'][0, :], dtype=int)
    num_groups = len(age_groups)-1 if ageing else uni['cfg']['OpDisc']['number_of_groups']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]
    time = uni['data/OpDisc/nw/opinion'].coords['time'].data

    #figure setup ..............................................................
    figure, axs = setup_figure(uni['cfg'], plot_name='opinion')
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    # data analysis and plotting................................................
    if to_plot == 'by_group':
        #get opinions by group
        to_plot = np.zeros((num_bins, num_groups))
        data_by_groups = data_by_group(opinions, groups, group_list, val_range,
                                       num_bins, ageing=ageing)

        #calculate a histogram of the opinion distribution at each time step
        for k in range(num_groups):
            counts, _ = np.histogram(data_by_groups[k][time_idx], bins=num_bins,
                                         range=val_range)
            to_plot[:, k] = counts[:]

        #get pretty labels
        if ageing:
            labels = [f"Ages {group_list[_]}-{group_list[_+1]}" for _ in range(num_groups)]
            max_age = np.amax(groups)
            if (age_groups[-1]>=max_age):
                labels[-1]=f"Ages {group_list[-2]}+"
        else:
            labels = [f"Group {_+1}" for _ in group_list]
        X = pd.DataFrame(to_plot[:, :], columns=labels)
        X.plot.bar(stacked=True, ax=hlpr.ax, legend=False, rot=0, **plot_kwargs)
        hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                       ncol=num_groups, fontsize='xx-small')
        hlpr.ax.set_xticks([i for i in np.linspace(0, num_bins-1, 11)])
        hlpr.ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


    elif to_plot == 'overall':
        hlpr.ax.hist(opinions[time_idx][:], bins=num_bins, alpha=1, **plot_kwargs)

    elif to_plot == 'discriminators':
        discriminators = uni['data/OpDisc/nw/discriminators']
        mask = np.empty(opinions.shape, dtype=bool)
        mask[:,:] = (discriminators == 0)
        ops_disc = np.ma.MaskedArray(opinions, mask)
        ops_nondisc = np.ma.MaskedArray(opinions, ~mask)
        hlpr.ax.hist(ops_disc[time_idx].compressed()[:], bins=num_bins, alpha=0.5, **plot_kwargs, label='disc')
        hlpr.ax.hist(ops_nondisc[time_idx].compressed()[:], bins=num_bins, alpha=0.5, **plot_kwargs, label='non-disc')
        hlpr.ax.hist(opinions[time_idx][:], bins=num_bins, alpha=1, **plot_kwargs, histtype='step')
        hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                       ncol=num_groups, fontsize='xx-small')
        hlpr.ax.set_xlim(val_range[0], val_range[1])

    hlpr.ax.text(0.02, 0.97, f'step {time[time_idx]}', transform=hlpr.ax.transAxes,
                 fontsize ='xx-small')
