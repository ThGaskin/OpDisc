import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utopya import DataManager, UniverseGroup
from utopya.plotting import UniversePlotCreator, PlotHelper, is_plot_func

from .data_analysis import data_by_group, find_const_vals, find_extrema
from .tools import setup_figure

# Get a logger
log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator,
              supports_animation=False, helper_defaults=dict(
                set_labels=dict(x="User opinion", y="Time")))
def group_avg(dm: DataManager, *,
              uni: UniverseGroup,
              hlpr: PlotHelper,
              age_groups: list=[10, 20, 40, 60, 80],
              num_bins: int=100,
              title: str=None,
              val_range: tuple=(0, 1)):
    """This function plots the average opinion of each group over time.

    Arguments:
       age_groups (list): the age binning to be plotted for the 'ageing' model
       num_bins (int, optional): binning size for the histogram
       title (str, optional): custom title for the plot
       val_range (tuple, optional): binning range for the histogram

    Raises:
        TypeError: if the 'age_groups' list does not contain at least two
            entries
    """
    if len(age_groups)<2:
        raise TypeError("'age_groups' list must contain at least 2 entries!")

    #figure setup ..............................................................
    figure, axs = setup_figure(uni['cfg'], plot_name='group_avg', title=title)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #get data ..................................................................
    ageing = True if uni['cfg']['OpDisc']['mode'] == 'ageing' else False
    opinions = uni['data/OpDisc/nw/opinion']
    if ageing:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'], dtype=int)
    else:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'][0, :], dtype=int)
    num_groups = len(age_groups)-1 if ageing else uni['cfg']['OpDisc']['number_of_groups']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]
    time_steps = opinions['time'].size
    time = np.asarray(opinions['time'].data)
    hlpr.ax.set_xlim(0, 1)
    hlpr.ax.set_ylim(time[-1], time[0])

    #data analysis..............................................................
    #calculate mean opinion and std of each group
    means = np.zeros((time_steps, num_groups))
    stddevs = np.zeros_like(means)
    data_by_groups = data_by_group(opinions, groups, group_list, val_range, num_bins,
                                   ageing=ageing)
    for k in range(num_groups):
        for t in range(time_steps):
            if len(data_by_groups[k][t])==0:
                #empty slices may occur if certain age groups are not present
                #for a period of time
                continue
            means[t, k] = np.mean(data_by_groups[k][t])
            stddevs[t, k] = np.std(data_by_groups[k][t])

    #plotting...................................................................
    #get pretty labels
    if ageing:
        labels = [f"Ages {group_list[_]}-{group_list[_+1]}" for _ in range(num_groups)]
        max_age = np.amax(groups)
        if (age_groups[-1]>=max_age):
            labels[-1]=f"Ages {group_list[-2]}+"
    else:
        labels = [f"Group {_+1}" for _ in range(num_groups)]

    #plot mean opinion with std as errorbar
    for i in range(num_groups):
        hlpr.ax.errorbar(means[:, i], time, xerr=stddevs[:, i], lw=2, alpha=0.8,
                       elinewidth=25./time_steps, label=labels[i], capsize=0, capthick=1)

    hlpr.ax.set_xticks(np.linspace(0, 1, 11), minor=False)
    hlpr.ax.xaxis.grid(True, which='major', lw=0.1)

    #temporary..................................................................
    #calculate the global mean and plot its turning points
    # mean_glob = pd.Series(np.mean(opinions, axis=1)).rolling(window=20).mean()
    # hlpr.ax.plot(mean_glob, time, lw=1, color='black', label='avg', zorder=num_groups+1)
    #
    # extremes = find_extrema(mean_glob, x=time)['max']
    # if extremes['y']:
    #      hlpr.ax.scatter(x=extremes['y'], y=extremes['x'], s=10, alpha=0.8, zorder=num_groups+2)
    #
    # constants = find_const_vals(mean_glob, time_steps, time=time, averaging_window=0.4, tolerance=0.01)
    #
    # if constants['t']:
    #     hlpr.ax.scatter(x=constants['x'], y=constants['t'], s=10, color='red', alpha=0.8, zorder=num_groups+2)
    hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                   ncol=num_groups+1, fontsize='xx-small')
