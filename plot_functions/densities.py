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
              plot_kwargs: dict=None,
              title: str=None,
              val_range: tuple=(0., 1.)):
    """Plots the density of user opinion over time.

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

    #data analysis and plotting................................................
    hlpr.ax.plot(data[:, :], data['time'], **plot_kwargs)
    hlpr.ax.set_xlim(val_range[0], val_range[1])
    hlpr.ax.set_ylim(data['time'][-1], 0)
