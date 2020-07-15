import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from utopya import DataManager, UniverseGroup
from utopya.plotting import UniversePlotCreator, PlotHelper, is_plot_func

from .tools import setup_figure

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator,
              supports_animation=False, helper_defaults=dict(
                set_labels=dict(x="Opinion", y="Step")))
def densities(dm: DataManager, *,
              uni: UniverseGroup,
              hlpr: PlotHelper,
              num_bins: int=100,
              title: str=None,
              val_range: tuple=(0., 1.)):
    """Plots the density of opinion clusters over time.

    Arguments:
        dm (DataManager): The data manager from which to retrieve the data
        uni (UniverseGroup): data group
        hlpr (PlotHelper): Description
        num_bins (int, optional): Binning of the histogram
        title (str, optional): Custom plot title
        val_range (tuple, optional): The range of the histogram
    """
    #figure layout..............................................................
    figure, axs = setup_figure(uni['cfg'], plot_name='densities', title=title)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)
    hlpr.select_axis(0, 1)

    #datasets...................................................................
    data = uni['data/OpDisc/nw/opinion']
    time_steps = data['time'].size

    #data analysis and plotting.................................................
    data_to_plot = np.zeros((time_steps, num_bins))
    for row in range(time_steps):
        counts_at_time, _ = np.histogram(data[row, :], range=val_range,
                                         bins=num_bins)
        data_to_plot[row, :] = counts_at_time/np.max(counts_at_time)
    hlpr.ax.set_xticks([i for i in np.linspace(0, num_bins-1, 11)])
    hlpr.ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    hlpr.ax.imshow(data_to_plot, cmap='BuGn', interpolation='bilinear',
                    aspect=num_bins/time_steps)
