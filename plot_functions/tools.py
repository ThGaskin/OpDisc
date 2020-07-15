"""Formatting and general utility functions for the OpDyn plots.
"""
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from typing import Tuple

log = logging.getLogger(__name__)

matplotlib.rcParams['mathtext.fontset']='stix'
matplotlib.rcParams['font.family']='serif'
rc('text', usetex=True)

#formatting ....................................................................
model_modes = {'ageing': 'directed conflict with ageing',
               'conflict_dir': 'directed conflict',
               'conflict_undir': 'undirected conflict',
               'isolated_1' : 'isolated discr. (type 1)',
               'isolated_2' : 'isolated discr. (type 2)',
               'reduced_s' : 'reduced susceptibility',
               'reduced_int_prob': 'reduced interaction probability'}

parameters = {'discriminators': r'Discriminator fraction $p_d$',
              'homophily_parameter': r'$p_\mathrm{hom}$',
              'life_expectancy': 'Life expectancy',
              'number_of_groups': 'Number of groups',
              'peer_radius': 'Peer radius',
              'susceptibility': r'Susceptibility $\mu$',
              'tolerance': r'Tolerance $\epsilon$'}

titles = {'bifurcation': r'\bf Bifurcation diagramme',
          'densities' : r'\bf Opinion clusters over time',
          'group_avg': r'\bf Average opinion by group',
          'group_avgs_anim': r'\bf Average opinion by group',
          'opinion': r'\bf Opinion distribution',
          'opinion_anim': r'\bf Opinion distribution over time',
          'op_groups' : r'\bf Opinion evolution by group'}

def convert_to_label(input) -> str:
    """Converts a parameter name from the cfg to a latex-readable string"""
    try:
        return parameters[input]
    except:
        log.warn(f"unrecognised parameter {input}!")
        return 'unrecognised parameter'

def mode(input) -> str:
    """Converts the model mode from the cfg to a latex-readable string"""
    try:
        return model_modes[input]
    except:
        log.warn(f"unrecognised model mode {input}!")
        return 'unrecognised mode'

def title_box(ax, cfg, *, plot_name: str, title: str=None, dim: str=None):
    """Returns a uniformly formatted title box

    Arguments:
        ax: the title axis
        cfg: the model config
        plot_name (str, optional)
        title (str, optional): the user-specified title
        dim (str, optional): the sweep dimension, if applicable
    """
    ax.axis('off')
    info = {'num_users': f"{cfg['OpDisc']['nw']['num_vertices']}",
            'num_steps': f"{cfg['num_steps']}"}
    for key in cfg['OpDisc'].keys():
        if key=='nw': continue
        info[key] = cfg['OpDisc'][key] if dim!=key else 'sweep'
    #title
    t = r'\bf {}'.format(title) if title else titles[plot_name]
    #subtitle
    st = f"Model: {mode(info['mode'])}"
    #model information
    if info['mode']=='ageing':
        sst = (f"{info['num_users']} users, " +
               f"life expectancy: {info['ageing']['life_expectancy']}, " +
               f"peer radius: {info['ageing']['peer_radius']} \n")
    else:
        sst = f"{info['num_users']} users, {info['number_of_groups']} groups \n"
    sst += (f"Numerical steps: {info['num_steps']} \n" +
            f"Susceptibility: {info['susceptibility']}, " +
            f"homophily parameter: {info['homophily_parameter']} \n" +
            f"Tolerance: {info['tolerance']}, extremism mode: {info['extremism']}")
    if info['mode']=='conflict_undir':
        sst += f"\n Discriminators: {info['discriminators']}"

    ax.text(0, 1.3, t, fontweight='bold', fontsize=20, verticalalignment='top',
                       horizontalalignment = 'left')
    ax.text(0, 1., st, fontsize=14, verticalalignment='top', horizontalalignment='left')
    ax.text(0, 0.0, sst, fontsize=10)

def setup_figure(cfg, *, plot_name: str, title: str=None, dim: str=None,
                 figsize: tuple=(8, 10), ncols: int=1, nrows: int=2,
                 height_ratios: list=[1, 6], width_ratios: list=[1],
                 gridspec: list=[(0, 0), (1, 0)]):
    """Sets up the figure and plots the title axis

    Arguments:
        cfg: the model cfg
        plot_name (str)
        title (str, optional): custom plot title
        dim (str, optional): sweep dimension
        figsize (tuple): figure size
        ncols (tuple): number of gridspec columns
        nrows (tuple): number of gridspec rows
        height_ratios (list)
        width_ratios (list)
        gridspec (list): the gridspec layout; list of tuples of ints or slices
    """
    figure = plt.figure(figsize=figsize)
    gs = figure.add_gridspec(ncols=ncols, nrows=nrows, height_ratios=height_ratios, width_ratios=width_ratios, hspace=0.2)
    axs = []
    for item in gridspec:
        axs.append([figure.add_subplot(gs[item])])
    title_box(axs[0][0], cfg, plot_name=plot_name, title=title, dim=dim)

    return figure, axs

#utility functions .............................................................
def R_p_factors(r, n) -> Tuple[float, float]:
    """Calculates the R_p prefactors for the various discrimination models.

    Arguments:
        r: number of vertices
        n: number of groups

    Returns:
        P: the interaction probability for the 'reduced interaction probability'
            mode
        Q: the interaction probability for the 'isolated discrimination' and
            'reduced susceptibility' modes
    """
    P = 1./((n-1)*(r-1))*((r/(2*(n-1))-1.)+((n-2)*(r/(n-1)-1.)))
    Q = 0.5*(1+2*(n-2))/(1.-1/(2*(n-1))+n-2-(n-2)/(n-1))

    return P, Q

def R_p(p_hom, n, P, Q, mode) -> list:
    """Calculates the R_p values for a given list of homophily parameters.

    Arguments:
        p_hom (list): the list of homophily parameters
        n (int): the number of groups
        P (double): the P-factor
        Q (double): the Q-factor
        mode (str): the model mode
    """
    if mode=="reduced_int_prob":
        return (P/(1-P)+p_hom)/(1.-p_hom)
    elif mode=="isolated_1" or mode=="reduced_s":
        return Q/((n-1)*(1-p_hom))
    elif mode=="isolated_2":
        return Q/((n-1)*(1-p_hom)**2)
    else:
        log.warn(f"R_p factors not applicable to mode {mode}!")
        return [-1]*len(p_hom)

def find_extremes(data, *, x=None) -> dict:
    """Returns a list of extrema of first ('osc') and second ('const') order.
    If an array of x values is passed, the x-coordinates of the extrema are also
    added.

    Arguments:
        data
        x (array, optional): x values
    """

    def find_root(i: int) -> float:
        """Estimates the zero between two passed function values with different
        signs by linear interpolation
        """
        x_1 = x[i+2]
        x_2 = x[i+3]
        y_1 = df[i]
        y_2 = df[i+1]
        return x_1 - y_1*(x_2-x_1)/(y_2-y_1) if y_2!=y_1 else 0.5*(x_2-x_1)

    res = {'const': {'x': [], 'y': []}, 'osc': {'x': [], 'y': []}}
    df = 0.5*(np.diff(data)[1:]+np.diff(data)[:-1]) #first derivative
    ddf = 0.5*(np.diff(df)[1:]+np.diff(df)[:-1]) #second derivative
    #df and ddf must have same length. they are shifted up by two wrt the data
    #array.
    df = df[1:-1]

    prev = -1
    for i in range(len(df)-1):
        if np.isnan(df[i]):
            #depending on the rolling averaging window, the first few
            #entries of data will be nans; skip them automatically
            continue
        if (abs(df[i])<1e-5) or np.sign(df[i])!= np.sign(df[i+1]):
            y_0 = (np.maximum(data[i+2], data[i+3]) if df[i]>0
                   else np.minimum(data[i+2], data[i+3]))
            #found local extremum: append x and y values
            if x is not None:
                x_0 = find_root(i)
            if ((res['const']['y']!=[] and abs(res['const']['y'][-1]-y_0)<0.01) or
                (res['osc']['y']!=[] and abs(res['osc']['y'][-1]-y_0)<0.01) or
                (ddf[i]==0 or np.sign(ddf[i])!=np.sign(ddf[i+1]))):
                #if second derivative is zero: constant value
                #if deviation from previous value small: random fluctuation
                #around constant value
                res['const']['y'].append(y_0)
                if x is not None:
                    res['const']['x'].append(x_0)
                prev = 0
            else:
                #if the previous entry was constant but the current entry an
                #oscillation, re-classify the last extremum as an oscillation also
                if prev==0:
                    res['osc']['y'].append(res['const']['y'].pop())
                    if x is not None:
                        res['osc']['x'].append(res['const']['x'].pop())
                res['osc']['y'].append(y_0)
                if x is not None:
                    res['osc']['x'].append(x_0)
                prev = 1

    return res

def data_by_group(data, groups, group_list, val_range: tuple=(0., 1.),
                 num_bins: int=100, *, ageing: bool) -> list:
    """For a given input of opinion values and group labels, this function
    sorts the opinions by groups and outputs the opinions of each group over time.

    Arguments:
        data (ndarray): the opinion dataset
        groups (ndarray): group label dataset
        group_list (list): list of groups to sort by
        val_range (tuple): range for the histogram binning
        num_bins (int): number of bins for the histogram
        ageing: (bool): whether the list of groups represents age intervals

    Raises:
        ValueError: if the dimension of the group labels is greater than two
    """
    data = np.asarray(data)
    groups = np.asarray(groups)
    if groups.ndim>2:
        raise ValueError(f"Invalid array dimension {groups.ndim}! Group label"
              " array must have dimension<3.")
    time_steps = data.shape[0]
    start, stop = val_range

    if groups.ndim==2:
        #check if all groups are covered by the given  groups; if not,
        #add maximum groups number to make sure they are.
        #This is necessary for pd.cut
        max_group_number = np.amax(groups)
        extended_groups = False
        if max_group_number>=group_list[-1]:
            extended_groups = True
            group_list.append(max_group_number+1)

        num_groups = len(group_list)-1 if ageing else len(group_list)

        data_by_group = [[[] for _ in range(time_steps)] for k in range(num_groups)]
        for t in range(time_steps):
            m = [[] for _ in range(num_groups)]
            group_bins = pd.cut(groups[t, :], group_list, labels=False,
                                  include_lowest=True)
            for i in range(data.shape[1]):
                m[group_bins[groups[t, i]]].append(data[t, i])
            for k in range(len(m)):
                data_by_group[k][t] = m[k]

        if extended_groups:
            data_by_group.pop(-1)
            group_list.pop(-1)

    else:
        num_groups = len(group_list)
        data_by_group = [[[] for _ in range(time_steps)] for k in range(num_groups)]
        i = np.argsort(groups)
        data = data[:,i]
        groups = groups[i]
        idx_jumps = np.zeros(num_groups+1, dtype=int)
        idx_jumps[-1] = data.shape[1]-1
        j = 1
        for i in range(data.shape[1]-1):
            if(groups[i+1]>groups[i]):
                idx_jumps[j]=i+1
                j+=1
        for t in range(time_steps):
            for _ in range(num_groups):
                data_by_group[_][t]=data[t, idx_jumps[_]:idx_jumps[_+1]]

    return data_by_group
