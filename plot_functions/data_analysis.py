"""Data analysis tools for the OpDisc plots."""
import logging
import numpy as np
import pandas as pd
from typing import Tuple

log = logging.getLogger(__name__)

## -----------------------------------------------------------------------------
def data_by_group(dataset, groups, group_list, val_range: tuple=(0., 1.),
                  num_bins: int=100, *, ageing: bool, time_step: int=None) -> list:
    """For a given input of opinion values and group labels, this function
    sorts the opinions by groups and outputs the opinions of each group over time.

    Arguments:
        dataset (ndarray): the opinion dataset
        groups (ndarray): group label dataset
        group_list (list): list of groups to sort by
        val_range (tuple): range for the histogram binning
        num_bins (int): number of bins for the histogram
        ageing: (bool): whether the list of groups represents age intervals
        time_step (int, optional): if the data array is 1d, the time step of the
            data array considered can be passed to ensure the group labels are
            also chosen from that time step

    Raises:
        ValueError: if the dimension of the group labels is greater than two
        ValueError: if a single time step is specified but the
    """
    #prepare data...............................................................
    data = np.asarray(dataset)
    #if a specific time step is considered (eg. the last one) only the opinion
    #value at that time step are relevant
    if data.ndim==2 and time_step:
        data = np.asarray([dataset[time_step]])
    #for certain runs, only a single time step will have been written (typically
    #the last one. In these cases, the data array is transformed into a 2d array
    #with one 'artifical' time step.
    elif data.ndim==1:
        data = np.asarray([dataset])
    time_steps = data.shape[0]
    start, stop = val_range

    #prepare group labels.......................................................
    groups = np.asarray(groups)
    if groups.ndim>2:
        raise ValueError(f"Invalid array dimension {groups.ndim}! Group label"
              " array must have dimension<3.")

    #the group labels can be constant or change over time (ageing model), and
    #they can also either represent group numbers or age bins. We must accout
    #for each of the three cases (1d and group label, 1d and age, 2d and age).
    #Case 1 group labels are time-dependent or represent age groups.............
    if ageing:
        #if a specific time step is considered (eg. the last one) only the group
        #labels at that time step are relevant in the case of temporally changing
        #labels
        if groups.ndim==2 and time_step:
            groups = np.asarray([groups[time_step]])

        #If the group labels are already 1d but still represent age groups, the
        #group label array is transformed into a 2d array with one 'artifical'
        #time step (as in the dataset above)
        elif groups.ndim==1:
            groups = np.asarray([groups])

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

        #if the age range was artificially extended, remove this additional
        #age bin
        if extended_groups:
            data_by_group.pop(-1)
            group_list.pop(-1)

    #group labels are constant and do not represent age labels..................
    else:
        #subspace selection
        if groups.ndim==2:
            groups = groups[0]
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

## -----------------------------------------------------------------------------
def get_means_stddevs(data, groups, group_list, *,
                      ageing: bool, time_step: int=None) -> Tuple[list, list]:
        """Returns the mean and standard deviation of the opinion distribution
        of each group for a single time step.

        Arguments:
            data (array or list, 1d): the opinion data for a single time step
            groups (array or list, 1d or 2d): the group labels of the users. If
                the group list is 2d and no time_step argument is passed, the
                group labels of the first time step will be automatically chosen
            group_list (array): the list by which to sort the data into groups
            ageing (bool): whether or not the groups represent age bins
            time_step (int, optional): the specific time frame considered.

        Returns:
            m (list): the means of each group
            s (list): the stddev of each group

        Raises:
            Warning: if two-dimensional group labels are passed without a specific
               time step.
        """
        if groups.ndim==2 and time_step is None:
            log.warn("Group labels are two-dimensional, yet you have not specified"
            " the specific time step of the dataset. The first time step will"
            " automatically be chosen. Pass a 'time_step' if you wish to specify"
            " the time step!")

        data_by_groups = data_by_group(data, groups, group_list, ageing=ageing,
                                       time_step=time_step)
        m = [abs(np.mean(data_by_groups[n])-0.5) for n in range(len(data_by_groups))]
        s = [np.std(data_by_groups[n]) for n in range(len(data_by_groups))]

        return m, s

## -----------------------------------------------------------------------------
def absolute_area(data, *, window: int=10) -> float:
    """Returns the absolute area (unsigned) under the mean curve minus 0.5.

    Arguments:
        data: the opinion dataset
        window (int, optional): the smoothing window for the rolling average

    Returns:
        A (float): the area (>=0)
    """
    mean_glob = pd.Series(np.mean(data, axis=1)).rolling(window=window).mean()
    means = np.abs(mean_glob-0.5)
    A = np.sum(means, axis=0)

    return A

## -----------------------------------------------------------------------------
def area(data, *, window: int=10) -> float:
    """Returns the area (signed) under the mean curve minus 0.5.

    Arguments:
        data: the opinion dataset
        window (int, optional): the smoothing window for the rolling average
    Returns:
        A (float): the area
    """
    mean_glob = pd.Series(np.mean(data, axis=1)).rolling(window=window).mean()
    means = mean_glob-0.5
    A = np.sum(means, axis=0)

    return np.abs(A)

## -----------------------------------------------------------------------------
def difference_of_extreme_means(mv_data, x, y, groups, group_list, *,
                                ageing: bool, group_1: int=0, group_2: int=-1, time_step: int=None):
    """Returns the difference of the means of the two groups specified.

    Arguments:
       mv_data: the multiverse data
       x (str): the first sweep dimension
       y (str): the second sweep dimension
       groups (array, 1d or 2d): the group labels of the users
       group_list (list): the list by which to sort user groups
       group_1 (int): the first group to plot
       group_2 (int): the second group to plot

    Returns:
       res (array): 2d array with resulting plot values
    """
    res = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))

    if 'number_of_groups' in mv_data.coords and len(mv_data.coords['number_of_groups'] > 0):
        if group_1!=0 or group_2!=-1:
            raise ValueError("When sweeping over the number of groups you"
             "can select to plot the difference between the means of the outer "
             "groups, ie. 1 and N")

        param2 = y if x=='number_of_groups' else x
        x = param2
        y = 'number_of_groups'
        res = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))
        for n in range(len(mv_data.coords[y])):
            num_groups = mv_data.coords[y].data[n]
            group_list = [_ for _ in range(num_groups)]
            groups = np.asarray(mv_data['group_label'][{x: 0, y: n, 'time': 0}], dtype=int)
            for i in range(len(mv_data.coords[x])):
                data = np.asarray(mv_data[{y: n, x: i, 'time': -1}]['opinion'])
                data_by_groups = data_by_group(data, groups, group_list, ageing=False)
                mean_0 = np.mean(data_by_groups[group_1])
                mean_1 = np.mean(data_by_groups[group_2])
                res[n, i] = mean_1-mean_0
    else:
        for param1 in range(len(mv_data.coords[x])):
            for param2 in range(len(mv_data.coords[y])):
                data = np.asarray(mv_data[{x: param1, y: param2, 'time':-1}]['opinion'])
                data_by_groups = data_by_group(data, groups, group_list,
                                                 ageing=ageing, time_step=time_step)
                mean_0 = np.mean(data_by_groups[group_1])
                mean_1 = np.mean(data_by_groups[group_2])
                res[param2, param1] = mean_1-mean_0

    return res

## -----------------------------------------------------------------------------
def find_const_vals(data, time_steps, *, time=None, averaging_window: float=0.3,
                    tolerance: float=0.01) -> dict:
    """Returns a list of constant values of an array.

    Arguments:
        data
        x (array, optional): x values
        averaging_window: the length of time (as a fraction of the total time)
            over which the data must remain constant
        tolerance: the value (in absolute) within which the data is allowed to
            fluctuate
    Returns:
        res: a dictionary containing x and y values of the constants
    """
    if averaging_window<0 or averaging_window>1:
        raise ValueError("Averaging window must be between 0 and 1!")
    if tolerance<0 or tolerance>1:
        raise ValueError("Tolerance must be between 0 and 1!")

    def is_const_val(t, window, data, tol):
        b=(abs(data[t]-data[t-window])<=tol)
        p = t-1
        ref_pt = t-window
        while b:
            b=(abs(data[p]-data[ref_pt])<=tol)
            if p==ref_pt+1:
                break
            p-=1
        return b

    res = {'t': [], 'x': []}
    l = int(averaging_window * time_steps)
    for i in range(l, time_steps):
        if is_const_val(i, l, data, tolerance):
            if time is not None:
                res['t'].append(time[int(i-l/2)])
            res['x'].append(data[int(i-l/2)])

    return res

## -----------------------------------------------------------------------------
def find_extrema(data, *, x=None) -> dict:
    """Returns a list of extrema of first order whose second derivative is not
    zero. If an array of x values is passed, the x-coordinates of the extrema are
    also added.

    Arguments:
        data
        x (array, optional): x values

    Returns:
        res: a dictionary containing extreme values sorted by minimum and maximum
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

    res = {'max': {'x': [], 'y': []}, 'min': {'x': [], 'y': []}}
    df = 0.5*(np.diff(data)[1:]+np.diff(data)[:-1]) #first derivative
    ddf = 0.5*(np.diff(df)[1:]+np.diff(df)[:-1]) #second derivative
    #df and ddf must have same length. they are shifted up by two wrt the data
    #array.
    df = df[1:-1]

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
            if ((res['max']['y']!=[] and abs(res['max']['y'][-1]-y_0)<0.01) or
                (res['min']['y']!=[] and abs(res['min']['y'][-1]-y_0)<0.01) or
                (ddf[i]==0 or np.sign(ddf[i-1])!=np.sign(ddf[i+1]))):
                continue
            else:
                if ddf[i]>0:
                    res['min']['y'].append(y_0)
                    if x is not None:
                        res['min']['x'].append(x_0)
                elif ddf[i]<0:
                    res['max']['y'].append(y_0)
                    if x is not None:
                        res['max']['x'].append(x_0)

    return res

## -----------------------------------------------------------------------------
def get_absolute_area(mv_data, keys, *, dim: str) -> Tuple[list, list]:
    """Returns a list of absolte areas (unsigned) and the stddevs of each value
    for a given sweep parameter.

    Arguments:
        mv_data (xdarray): the multiverse data
        keys (dict): the keys with the subspace selection
        dim (str): the sweep dimension

    Returns:
        plot_data (list): a list of area of length (sweep parameter dimension)
        err (list): a list of standard deviations
    """
    data_to_plot = {}
    for i in range(len(mv_data.coords[dim].data)):
        data_to_plot[mv_data.coords[dim].data[i]] = []
    if 'seed' in mv_data.coords and len(mv_data.coords['seed'].data)>1:
        for i in range(len(mv_data.coords[dim])):
            for j in range(len(mv_data.coords['seed'])):
                keys[dim] = i
                keys['seed'] = j
                data = np.asarray(mv_data[keys]['opinion'])
                res = absolute_area(data)
                data_to_plot[mv_data.coords[dim].data[i]].append(res)
    else:
        for i in range(len(mv_data.coords[dim])):
            keys[dim]=i
            data = np.asarray(mv_data[keys]['opinion'])
            res = absolute_area(data)
            data_to_plot[mv_data.coords[dim].data[i]].append(res)

    plot_data = []
    err = []
    for key in data_to_plot.keys():
        plot_data.append(np.mean(data_to_plot[key]))
        err.append(np.std(data_to_plot[key]))

    return plot_data, err

## -----------------------------------------------------------------------------
def get_area(mv_data, keys, *, dim: str) -> Tuple[list, list]:
    """Returns a list of areas (signed) and the stddevs of each value for a given
    sweep parameter.

    Arguments:
        mv_data (xdarray): the multiverse data
        keys (dict): the keys with the subspace selection
        dim (str): the sweep dimension

    Returns:
        plot_data (list): a list of area of length (sweep parameter dimension)
        err (list): a list of standard deviations
    """
    data_to_plot = {}
    for i in range(len(mv_data.coords[dim].data)):
        data_to_plot[mv_data.coords[dim].data[i]] = []
    if 'seed' in mv_data.coords and len(mv_data.coords['seed'].data)>1:
        for i in range(len(mv_data.coords[dim])):
            for j in range(len(mv_data.coords['seed'])):
                keys[dim] = i
                keys['seed'] = j
                data = np.asarray(mv_data[keys]['opinion'])
                res = area(data)
                data_to_plot[mv_data.coords[dim].data[i]].append(res)
    else:
        for i in range(len(mv_data.coords[dim])):
            keys[dim]=i
            data = np.asarray(mv_data[keys]['opinion'])
            res = area(data)
            data_to_plot[mv_data.coords[dim].data[i]].append(res)

    plot_data = []
    err = []
    for key in data_to_plot.keys():
        plot_data.append(np.mean(data_to_plot[key]))
        err.append(np.std(data_to_plot[key]))

    return plot_data, err

## -----------------------------------------------------------------------------
def means_stddevs_by_group(mv_data, group_list, dim, keys, *, mode: str,
                           ageing: bool, num_groups: int, which: str,
                           time_step: int=None) -> Tuple[list, list]:
    """Returns a list of means or stddevs of each group opinion distribution for
    a given sweep configuration.

    Arguments:
        mv_data (xdarray): the multiverse data
        group_list (list): the group list
        dim (str): the sweep parameter
        keys (dict): the keys containing all subspace selections
        mode (str): the model mode
        ageing (bool): whether or not the group_list represents age bins
        num_groups (int): the number of groups we are considering (length of
            the group_list except in the 'ageing' model, where it the length
            of the group_list-1)
        which (str): whether to return the means or stddevs (must be either
            'means' or 'stddevs')
        time_step (int, optional): which time step is considered; in the case of
            two dimensional (ie. time-dependent) group labels, the group labels
            for that time step are used.
    Returns:
        data_to_plot (list): the values to plot
        err (list): the stddev of each value
    """
    means = {}
    stddevs = {}
    for _ in (range(num_groups)):
        means[_] = {}
        stddevs[_] = {}
        for i in range(len(mv_data.coords[dim].data)):
            means[_][mv_data.coords[dim].data[i]] = []
            stddevs[_][mv_data.coords[dim].data[i]] = []

    if 'seed' in mv_data.coords and len(mv_data.coords['seed'].data)>1:
        for i in range(len(mv_data.coords[dim])):
            for j in range(len(mv_data.coords['seed'])):
                keys[dim] = i
                keys['seed'] = j
                keys['time'] = -1
                data = np.asarray(mv_data[keys]['opinion'])
                #for all modes except ageing, the group labels do not change.
                #we can thus extract the group labels from the first time step
                if mode != 'ageing':
                    keys['time']=0
                groups = np.asarray(mv_data[keys]['group_label'], dtype=int)
                m, s = get_means_stddevs(data, groups, group_list, ageing=ageing,
                                         time_step=time_step)
                for _ in range(num_groups):
                    means[_][mv_data.coords[dim].data[i]].append(m[_])
                    stddevs[_][mv_data.coords[dim].data[i]].append(s[_])
    else:
        for i in range(len(mv_data.coords[dim].data)):
            keys[dim] = i
            keys['time'] = -1
            data = np.asarray(mv_data[keys]['opinion'])
            #for all modes except ageing, the group labels do not change.
            #we can thus extract the group labels from the first time step
            if mode != 'ageing':
                keys['time']=0
            groups = np.asarray(mv_data[keys]['group_label'], dtype=int)
            m, s = get_means_stddevs(data, groups, group_list, ageing=ageing,
                                     time_step=time_step)
            for _ in range(num_groups):
                means[_][mv_data.coords[dim].data[i]].append(m[_])
                stddevs[_][mv_data.coords[dim].data[i]].append(s[_])

    data_to_plot = [[] for n in range(num_groups)]
    err = [[] for n in range(num_groups)]

    for n in range(num_groups):
        if which == 'means':
            for key in means[n].keys():
                data_to_plot[n].append(np.mean(means[n][key]))
                err[n].append(np.std(means[n][key]))
        elif which =='stddevs':
            for key in stddevs[n].keys():
                data_to_plot[n].append(np.mean(stddevs[n][key]))
                err[n].append(np.std(stddevs[n][key]))

    return data_to_plot, err

## -----------------------------------------------------------------------------
def avgs_with_changing_groups(mv_data, x, y, *, which: str, time_step: int=-1):
    """Returns a two-dimensional array of the average distance of the group means
    at a single time_step to 0.5 if the number of groups is a sweep parameter.
    The absolute difference of each group mean to 0.5 is calculated,
    and its average returned. Only relevant to modes that are not 'ageing'.

    Arguments:
       mv_data (xdarray): the multiverse dataset
       x (str): the first sweep dimension
       y (str): the second sweep dimension
       which (str): whether to calculate means or stddevs
       time_step (int, optional): which time step is considered
     Returns:
        res (array): the resulting 2d array of values
    """
    param2 = y if x=='number_of_groups' else x
    x = param2
    y = 'number_of_groups'
    res = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))

    w = 0 if which=='means' else 1

    if 'seed' in mv_data.coords and len(mv_data.coords['seed'].data > 0):
        for n in range(len(mv_data.coords[y])):
            num_groups = mv_data.coords[y].data[n]
            group_list = [_ for _ in range(num_groups)]
            for seed in range(len(mv_data.coords['seed'])):
                groups = np.asarray(mv_data['group_label'][{x: 0, y: n, 'time': 0, 'seed': seed}], dtype=int)
                for i in range(len(mv_data.coords[x])):
                    data = mv_data['opinion'][{y: n, x: i, 'time': time_step, 'seed': seed}]
                    d = get_means_stddevs(data, groups, group_list, ageing=False,
                                          time_step=-1)[w]
                    res[n, i]+=np.mean(d, axis=0)/len(mv_data.coords['seed'])
    else:
        for n in range(len(mv_data.coords[y])):
            num_groups = mv_data.coords[y].data[n]
            group_list = [_ for _ in range(num_groups)]
            groups = np.asarray(mv_data['group_label'][{x: 0, y: n, 'time': 0}], dtype=int)
            for i in range(len(mv_data.coords[x])):
                data = mv_data[{y: n, x: i, 'time': time_step}]['opinion']
                d = get_means_stddevs(data, groups, group_list, ageing=False,
                                      time_step=time_step)[w]
                res[n, i] = np.mean(d, axis=0)

    return res

## -----------------------------------------------------------------------------
def avg_of_means_stddevs(mv_data, x, y, groups, group_list, keys, mode,
                            num_groups, *, which: str, ageing: bool, time_step: int=-1):
    """Returns a two-dimensional array of the average distance of the group means
    at a single time_step to 0.5. The absolute difference of each group mean to
    0.5 is calculated, and its average returned.

    Arguments:
       mv_data (xdarray): the multiverse dataset
       x (str): the first sweep dimension
       y (str): the second sweep dimension
       groups (list): the group labels
       group_list (list): the list of possible groups
       keys (dict): the subspace selection
       mode (str): the model mode
       num_groups (int): number of groups; is equal to the length of the group
           list for every model except ageing, where it is the length of the
           group list minus 1
       which (str): whether to return means or stddevs
       ageing (bool): whether or not the group list reflect age bins
       time_step (int, optional): which time step is considered

     Returns:
        res (array): the resulting 2d array of values

     Raises:
        ValueError: if incorrect keyword 'which' passed

    """
    if which not in ['means', 'stddevs']:
        raise ValueError("Error: 'which' key must be one of 'means' or 'stddevs!'")

    #special case: we are sweeping over the number of groups. In this case,
    #the number of groups and group_list changes, and we can no longer cut the
    #groups the same way for each sweep. In this case, we need calculate the
    #number of groups and group labels anew for each bin
    if 'number_of_groups' in mv_data.coords and len(mv_data.coords['number_of_groups']>0):
        return avgs_with_changing_groups(mv_data, x, y, which=which)

    res = np.zeros((len(mv_data.coords[y]), len(mv_data.coords[x])))
    #regular case: number of groups are constant. In this case, we only need to
    #differentiate between the seed key being present or not.
    if 'seed' in mv_data.coords and len(mv_data.coords['seed'].data > 0):
        for param1 in range(len(mv_data.coords[x])):
            keys.update({'time': -1, x: param1})
            data = means_stddevs_by_group(mv_data, group_list, y, keys,
                                 mode=mode, ageing=ageing, num_groups=num_groups,
                                 which=which, time_step=time_step)[0]
            res[:, param1] = np.mean(data, axis=0)

    else:
        w = 0 if which=='means' else 1
        for param1 in range(len(mv_data.coords[x])):
            for param2 in range(len(mv_data.coords[y])):
                data = mv_data[{x: param1, y: param2, 'time': time_step}]['opinion']
                d = get_means_stddevs(data, groups, group_list, ageing=ageing,
                                         time_step=time_step)[w]
                res[param2, param1] = np.mean(d, axis=0)

    return res
