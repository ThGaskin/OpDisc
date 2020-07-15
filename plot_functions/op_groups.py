import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd

from utopya import DataManager, UniverseGroup
from utopya.plotting import UniversePlotCreator, PlotHelper, is_plot_func

from .tools import data_by_group, setup_figure

log = logging.getLogger(__name__)
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator,
              supports_animation=True, helper_defaults=dict(
                set_labels=dict(x=r"User opinion", y=r"Group size")))
def op_groups(dm: DataManager, *,
              uni: UniverseGroup,
              hlpr: PlotHelper,
              age_groups: list=[10, 20, 40, 60, 80],
              num_bins: int=100,
              time_idx: int=None,
              title: str=None,
              val_range: tuple=(0., 1.)):
    """Plots an animated stacked histogram of the opinion distribution of
       each group.

    Arguments:
        age_groups (list): The age intervals to be plotted in the final_ax
            distribution plot for the 'ageing' mode.
        num_bins(int): Binning of the histogram
        time_idx (int, optional): Only plot one single frame (eg. last frame)
        title (str, optional): Custom plot title
        val_range(int, optional): Value range of the histogram

    Raises:
        TypeError: if the 'age_groups' list does not contain at least two
            entries
    """

    if len(age_groups)<2:
        raise TypeError("'age_groups' list must contain at least 2 entries!")

    #figure setup..............................................................
    figure, axs = setup_figure(uni['cfg'], plot_name='op_groups', title=title)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #datasets ..................................................................
    ageing = True if uni['cfg']['OpDisc']['mode'] == 'ageing' else False
    if ageing:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'], dtype=int)
    else:
        groups = np.asarray(uni['data/OpDisc/nw/group_label'][0, :], dtype=int)
    num_groups = len(age_groups)-1 if ageing else uni['cfg']['OpDisc']['number_of_groups']
    group_list = age_groups if ageing else [_ for _ in range(num_groups)]
    opinions = uni['data/OpDisc/nw/opinion']
    time = opinions.coords['time'].data
    time_steps = opinions.coords['time'].size

    #data analysis .............................................................
    to_plot = np.zeros((time_steps, num_bins, num_groups))
    data_by_groups = data_by_group(opinions, groups, group_list, val_range,
                                   num_bins, ageing=ageing)
    for t in range(time_steps):
        for k in range(num_groups):
            counts, _ = np.histogram(data_by_groups[k][t], bins=num_bins,
                                     range=val_range)
            to_plot[t, :, k] = counts[:]
    if ageing:
        labels = [f"Ages {group_list[_]}-{group_list[_+1]}" for _ in range(num_groups)]
        max_age = np.amax(groups)
        if (age_groups[-1]>=max_age):
            labels[-1]=f"Ages {group_list[-2]}+"
    else:
        labels = [f"Group {_+1}" for _ in group_list]
    if not time_idx:
        X = [pd.DataFrame(to_plot[_, :, :], columns=labels) for _ in range(time_steps)]
    else:
        X = pd.DataFrame(to_plot[time_idx, :, :], columns=labels)

    #plotting ..................................................................
    def update_data(stepsize: int=1):
        """Updates the data of the imshow objects"""
        if time_idx:
            log.info(f"Plotting discribution at time step {time[time_idx]} ...")
        else:
            log.info(f"Plotting animation with {opinions.shape[0]//stepsize} frames ...")
        next_frame_idx = 0
        if time_steps < stepsize:
            log.warn("Stepsize is greater than number of steps. "
                          "Continue by plotting fist and last frame.")
            stepsize=time_steps-1
        for t in range(time_steps):
            if t < next_frame_idx:
                continue
            hlpr.ax.clear()
            hlpr.ax.set_xlim(0, 1)
            if time_idx:
                t = time_idx
                im = X.plot.bar(stacked=True, ax=hlpr.ax, legend=False, rot=0)
            else:
                im = X[t].plot.bar(stacked=True, ax=hlpr.ax, legend=False, rot=0)
            time_text = hlpr.ax.text(0.02, 0.97, '', transform=hlpr.ax.transAxes,
                                     fontsize ='xx-small')
            time_text.set_text(f'step {time[t]}')
            hlpr.ax.legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                                           ncol=num_groups, fontsize='xx-small')
            hlpr.ax.set_xticks([i for i in np.linspace(0, num_bins-1, 11)])
            hlpr.ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            hlpr.ax.set_xlabel(hlpr.axis_cfg['set_labels']['x'])
            hlpr.ax.set_ylabel(hlpr.axis_cfg['set_labels']['y'])
            if time_idx:
                yield
                break
            next_frame_idx = t + stepsize
            yield
    hlpr.register_animation_update(update_data)
