"""Formatting and general utility functions for the OpDisc plots."""
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from matplotlib import rc

log = logging.getLogger(__name__)

matplotlib.rcParams['mathtext.fontset']='stix'
matplotlib.rcParams['font.family']='serif'
rc('text', usetex=True)

#formatting ....................................................................
model_modes = {
    'ageing': 'directed conflict with ageing',
    'conflict_dir': 'directed conflict',
    'conflict_undir': 'undirected conflict',
    'isolated_1' : 'isolated discr. (type 1)',
    'isolated_2' : 'isolated discr. (type 2)',
    'reduced_s' : 'reduced susceptibility',
    'reduced_int_prob': 'reduced interaction probability'
}

parameters = {
    'absolute_area': r'$\vert A^{+} \vert + \vert A^{-} \vert$',
    'area': r'$\vert A \vert $',
    'area_comp': r'$A$',
    'area_diff': r'$\vert A \vert - A$',
    'avg_of_means_diff_to_05': r'$\langle \vert \bar{\sigma}-0.5 \vert \rangle$',
    'avg_of_stddevs': r'$\langle \mathrm{var}(\bar{\sigma}) \rangle$',
    'discriminators': r'$p_d$',
    'homophily_parameter': r'$p_\mathrm{hom}$',
    'life_expectancy': 'Life expectancy',
    'mean_degree': r'$\bar{k}$',
    'means': r'$\langle \vert \bar{\sigma}-0.5 \vert \rangle$',
    'number_of_groups': 'N',
    'peer_radius': 'Peer radius',
    'stddevs': r'$\langle \mathrm{var}(\bar{\sigma}) \rangle$',
    'susceptibility': r'$\mu$',
    'time_scale': r'time scale',
    'tolerance': r'$\epsilon$'
}

titles = {
    'absolute_area': r'\bf Absolute area under means curve',
    'area': r'\bf Area under means curve',
    'area_comp': r'\bf Comparison of areas under means curve',
    'area_diff': r'\bf Difference of areas under means curve',
    'avg_of_means_diff_to_05': r'\bf Average of difference of means to 0.5',
    'avg_of_stddevs': r'\bf Average of standard deviations',
    'bifurcation': r'\bf Bifurcation diagramme',
    'densities' : r'\bf Opinion clusters over time',
    'extreme_means_diff': r'\bf Difference of means of groups 1 and N',
    'group_avg': r'\bf Average opinion by group',
    'group_avgs_anim': r'\bf Average opinion by group',
    'means': r'\bf Distribution means over time',
    'opinion': r'\bf Opinion distribution at single time step',
    'opinion_anim': r'\bf Opinion distribution over time',
    'op_groups' : r'\bf Opinion evolution by group',
    'stddevs': r'\bf Distribution variances over time',
}

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

def title_box(ax, cfg, *, plot_name: str, title: str=None, dim1: str=None, dim2: str=None):
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
        elif key=='ageing':
            info['ageing']={}
            for k in cfg['OpDisc'][key].keys():
                info['ageing'][k] = cfg['OpDisc']['ageing'][k] if k not in [dim1, dim2] else 'sweep'
        else:
            info[key] = cfg['OpDisc'][key] if key not in [dim1, dim2] else 'sweep'
    #title
    t = r'\bf {}'.format(title) if title else titles[plot_name]
    #subtitle
    st = f"Model: {mode(info['mode'])}"
    #model information
    if info['mode']=='ageing':
        sst = (f"{info['num_users']} users, " +
               f"life expectancy: {info['ageing']['life_expectancy']}, " +
               f"time scale: {info['ageing']['time_scale']}, "+
               f"peer radius: {info['ageing']['peer_radius']} \n")
    else:
        sst = f"{info['num_users']} users, number of groups: {info['number_of_groups']} \n"
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

def setup_figure(cfg, *, plot_name: str, title: str=None, dim1: str=None, dim2: str=None,
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
    #gs.update(left=0.,right=1,top=1,bottom=0.0,wspace=0.3,hspace=0.09)
    axs = []
    for item in gridspec:
        axs.append([figure.add_subplot(gs[item])])
    title_box(axs[0][0], cfg, plot_name=plot_name, title=title, dim1=dim1, dim2=dim2)

    return figure, axs

#utility functions .............................................................
def R_p(p_hom, n, mode, *, P: float=1., Q: float=1.) -> list:
    """Calculates the R_p values for a given list of homophily parameters.

    Arguments:
        p_hom (list): the list of homophily parameters
        n (int): the number of groups
        P (float, optional): the P-factor
        Q (float, optional): the Q-factor
        mode (str): the model mode

    Raises:
        Warning: if a mode is passed for which the R_p factors cannot be calculated

    Returns:
        R_p (list): list of R_p factors
    """
    if isinstance(p_hom, float) and p_hom==1:
        return np.inf
    if mode=="reduced_int_prob":
        return (P/(n-1)+p_hom)/(1.-p_hom)
    elif mode=="isolated_1" or mode=="reduced_s":
        return Q/((n-1)*(1-p_hom))
    elif mode=="isolated_2":
        return Q/((n-1)*(1-p_hom)**2)
    else:
        log.warn(f"R_p factors not applicable to mode {mode}!")
        return [-1]*len(p_hom)

def deduce_sweep_dimension(data, *, key_to_ignore: str='seed') -> str:
    """Tries to automatically deduce the sweep dimension if no dim argument
    is passed.

    Arguments:
        data: the multiverse data containing coords and dims
        key_to_ignore: a key that can be ignored even if it is a sweep dimension
        (typically the seed)

    Raises:
        ValueError: if too many sweep dimensions are present and dim cannot be
            inferred

    Returns:
        dim (str): the deduced sweep dimension
    """
    if len(data.dims)>3:
        have_key = False
        for key in data.dims.keys():
            if key not in ['vertex', 'time', key_to_ignore] and data.dims[key]>1:
                if have_key:
                    raise ValueError(f"Automatic sweep parameter deduction failed: "
                          "too many dimensions! Use 'subspace' to select specific"
                          " values for keys other than the desired sweep key!")
                else:
                    dim = key
                    have_key = True
        if not have_key:
            raise ValueError("No sweep parameter available.")
    else:
        have_key = False
        for key in data.dims.keys():
            if key=='time' or key=='vertex' or key==key_to_ignore:
                continue
            dim = key
            have_key = True
        if not have_key:
            raise ValueError("No sweep parameter available.")

    log.info(f"Deduced sweep dimension '{dim}'")

    return dim

def get_keys_cfg(mv_data, config, *, keys_to_ignore: list) -> Tuple[dict, dict]:
    """Modifies the multiverse keys to easily allow subspace selections to be
    treated the same ways as full multiverse sweeps. The function sets the index
    for the dataset selection for any subspace selections to 0, excluding the
    time and vertex keys, as well as any other specified keys. This function also
    modifies the cfg, such that the entry value of any non-sweep parameters
    is the subspace selection. This allows the plot information box to display
    the correct parameters for any given configuration.

    Arguments:
        mv_data (xdarray): the full multiverse data
        config (dict): the original parameter space configuration, containing all
            yaml tags and sweep values
        keys_to_ignore (list, optional): any keys to ignore in the modification
            process (typically the seed)

    Raises:
        ValueError: if the subspace selection is insufficient

    Returns:
        keys (dict): the dictionary of keys and corresponding subspace indices
        config (dict): the modified configuration file
    """
    keys = dict(zip(dict(mv_data.dims).keys(), [0]*len(mv_data.dims)))
    #vertex and the selected dimension will never be subspace selections,
    #and should not be fixed to a single entry
    for key in set(['vertex', 'time']+keys_to_ignore):
        keys.pop(key)

    #assert correct parameterspace dimensionality;
    #manually set any subspace selection parameters in the cfg;
    #for subspace selection: set value to 0 for access during data analysis
    for key in keys:
        if key=='seed' and len(mv_data.coords['seed'].data)>1:
            keys['seed'] = [_ for _ in range(len(mv_data.coords['seed'].data))]
            continue
        elif mv_data.dims[key]>1:
            raise ValueError(f"Too many dimensions! Use 'subspace' to "
                             "select specific values for keys other than "
                             f"{keys_to_ignore} and 'seed'!")
        if key in ['life_expectancy', 'peer_radius', 'time_scale']:
            config['OpDisc']['ageing'][key] = mv_data.coords[key].data[0]
        else:
            config['OpDisc'][key] = mv_data.coords[key].data[0]
        keys[key] = 0

    return keys, config
