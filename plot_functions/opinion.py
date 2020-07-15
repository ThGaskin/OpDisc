import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

from utopya import DataManager, UniverseGroup
from utopya.plotting import is_plot_func, UniversePlotCreator, PlotHelper

from .tools import setup_figure

#matplotlib.rcParams['mathtext.fontset']='stix'
#matplotlib.rcParams['font.family']='serif'
#rc('text', usetex=True)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator)
def opinion(dm: DataManager,
            *,
            hlpr: PlotHelper,
            uni: UniverseGroup,
            age_groups: list=[10, 20, 40, 60, 80],
            title: str=None):
    """Plots the init_axial and axs[2] opinion states. If the model mode is 'ageing',
    the opinions of the specified age groups are plotted, in all other cases
    the opinions of the groups are shown.

    Arguments:
        age_groups (list): The age intervals to be plotted in the axs[2]
            distribution plot for the 'ageing' mode.
        title (str, optional): Custom plot title

    Raises:
        TypeError: if the age groups have fewer than two entries (in which
        case no binning is possible)
    """
    if len(age_groups)<2:
        raise TypeError("'Age groups' list needs at least two entries!")

    #datasets...................................................................
    mode        = uni['cfg']['OpDisc']['mode']
    w           = -1 if mode=='ageing' else 0
    groups      = np.asarray(uni['data/OpDisc/nw/group_label'][w], dtype=int)
    opinions    = uni['data/OpDisc/nw/opinion'][[0, -1], :]
    num_groups  = uni['cfg']['OpDisc']['number_of_groups']
    max_age     = np.amax(groups)

    #figure layout..............................................................
    figure, axs = setup_figure(uni['cfg'], plot_name='opinion', title=title,
                               nrows=3, height_ratios=[1, 3, 3],
                               gridspec=[(0, 0), (1, 0), (2, 0)])
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)

    # data analysis and plotting................................................
    if mode=='ageing':
        #check if all ages are covered by the given age groups; if not,
        #add maximum age to make sure they are. This is necessary for pd.cut
        extended_age_range = False
        if (age_groups[-1]<max_age):
            age_groups.append(max_age+1)
            extended_age_range = True
        op_by_groups = [[] for _ in range(len(age_groups)-1)]
        age_bins = pd.cut(groups, age_groups, labels=False, include_lowest=True)
        for _ in range(opinions.shape[1]):
            op_by_groups[age_bins[_]].append(opinions[-1, _])
        if extended_age_range:
            op_by_groups.pop()
            age_groups.pop()

        labels = [f"Ages {age_groups[_]}-{age_groups[_+1]}" for _ in range(len(age_groups)-1)]
        if age_groups[-1]>=max_age:
            labels[-1]=f"Ages {age_groups[-2]}+"

        #plot overall distribution
        axs[1][0].hist(opinions[0, :], bins=100, alpha=1)

        #plot age group distributions
        for j in range(len(age_groups)-1):
            axs[2][0].hist(op_by_groups[j], bins=100, alpha=.5, range=(0, 1),
                          label=labels[j])

    else:
        labels=[f'Group {i+1}' for i in range(num_groups)]
        op_by_groups = [[[] for i in range(num_groups)] for _ in range(2)]
        for _ in range(opinions.shape[1]):
            op_by_groups[0][groups[_]].append(opinions[0, _])
            op_by_groups[-1][groups[_]].append(opinions[-1, _])

        #plot overall distribution
        axs[1][0].hist(opinions[0, :], bins=100, histtype='step', alpha=1,
                           color='slategray', linestyle='dotted', linewidth=0.8)

        #plot group distributions
        for j in range(num_groups):
            axs[1][0].hist(op_by_groups[0][j][:], bins=100, alpha=.5,
                                                  range=(0, 1), label=labels[j])
            axs[2][0].hist(op_by_groups[-1][j][:], bins=100, alpha=.5, range=(0, 1))

    #layout ....................................................................
    for plot in [axs[1][0], axs[2][0]]:
        plot.set_xlim(0, 1)
        plot.grid(axis='x')
        plot.set_ylabel('Group size')
    axs[2][0].set_xlabel('User opinion')

    #add text
    axs[1][0].text(0.01, 0.95, 'initial distribution', fontsize=10,
                                                    transform=axs[1][0].transAxes)
    axs[2][0].text(0.01, 0.95, 'final distribution', fontsize=10,
                                                   transform=axs[2][0].transAxes)
    if mode=='ageing':
        lines, labels = axs[2][0].get_legend_handles_labels()
        axs[1][0].legend(lines, labels, bbox_to_anchor=(1, 1.01), loc='lower right',
                                    fontsize='xx-small', ncol=len(age_groups)-1)
    else:
        axs[1][0].legend(bbox_to_anchor=(1, 1.01), loc='lower right',
                                           fontsize='xx-small', ncol=num_groups)
