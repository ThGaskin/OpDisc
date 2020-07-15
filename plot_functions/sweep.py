import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import os
import time
from scipy.signal import find_peaks

from dantro.tools import recursive_update
from utopya import DataManager, UniverseGroup
from utopya.plotting import is_plot_func, PlotHelper, MultiversePlotCreator
from matplotlib import rc

rc('text', usetex=True)
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex')
# -----------------------------------------------------------------------------

@is_plot_func(creator_type=MultiversePlotCreator)
def sweep( dm: DataManager,
           *,
           hlpr: PlotHelper,
           mv_data,
           plot_prop: str,
           dim: str,
           dim2: str=None,
           stack: bool=False,
           write_only: bool=False,
           no_errors: bool=True,
           data_name: str='opinion',
           bin_number: int=100,
           plot_kwargs: dict=None):
    """Plots a bifurcation diagram for one parameter dimension (dim)
        i.e. plots the chosen final distribution measure over the parameter,
        plots stacked one-dimensional plots in a single diagramme,
        or - if second parameter dimension (dim2) is given - plots the
        2d parameter space as a heatmap.

    Configuration:
        - use the `select/field` key to associate one or multiple datasets
        - change `data_name` if needed
        - choose the dimension `dim` (and `dim2`) in which the sweep was
          performed.

    Arguments:
        dm (DataManager): The data manager from which to retrieve the data
        hlpr (PlotHelper): Description
        mv_data (xr.Dataset): The extracted multidimensional dataset
        plot_prop (str): The quantity that is extracted from the
            data. Available are: ['number_of_peaks', 'localization',
            'max_distance', 'polarization', 'peaks_by_group', 'tipping_points']
        dim (str): The parameter dimension of the diagram
        dim2 (str, optional): The second parameter dimension of the diagram
        stack (bool, optional): Whether the plots for dim2 are stacked
            or extend to heatmap
        data_name (str, optional): Description
        bin_number (int, optional): default: 100
            number of bins for the discretization of the final distribution
        plot_kwargs (dict, optional): passed to the plot function

    Raises:
        TypeError: for a parameter dimesion higher than 5
            (and higher than 4 if not sweeped over seed or homophily/rejection parameters)
        ValueError: If 'data_name' data does not exist
    """
    if not dim in mv_data.dims:
        raise ValueError("Dimension `dim` not available in multiverse data."
                         " Was: {} with value: '{}'."
                         " Available: {}"
                         "".format(type(dim),
                                   dim,
                                   mv_data.coords))
    if len(mv_data.coords) > 5:
        raise TypeError("mv_data has more than two extra parameter dimensions."
                        " Are: {}. Chosen dim: {}. (Max: ['vertex', 'time', "
                        "'seed'] + 2)".format(mv_data.coords, dim))

    if (len(mv_data.coords) > 4) and ('seed' not in mv_data.coords) and ('homophily_parameter' not in mv_data.coords):
        raise TypeError("mv_data has more than two extra parameter dimensions."
                        " Are: {}. Chosen dim: {}. (Max: ['vertex', 'time', "
                        "'seed'] + 2)".format(mv_data.coords, dim))

    plot_kwargs = (plot_kwargs if plot_kwargs else {})

    # Default plot configurations
    plot_kwargs_default_1d = {'lw': 1.5, 'mec': None,
                              'capsize': 1, 'mew': .6, 'ms': 2}

    if no_errors:
        plot_kwargs_default_1d = {'ms': 2, 'lw': 1.5}

    plot_kwargs_default_2d = {'origin': 'lower', 'cmap': 'GnBu'}

    # Color palette
    global color_idx
    color_idx = 0

    colors=['gold', 'orange', 'cornflowerblue', 'navy', 'slategray',
            'darkorange', 'royalblue', 'peru', 'orangered',
            'darkred','indigo','saddlebrown','firebrick','black' ]

    # Get group labels
    if(plot_prop == 'tipping_points' or plot_prop=='peaks_by_group'):
        if (len(mv_data.coords)==3):
            groups = np.asarray(mv_data['group_label'][0, 0, :])
        elif (len(mv_data.coords)==4):
            groups = np.asarray(mv_data['group_label'][0, 0, 0, :])
        elif (len(mv_data.coords)==5):
            groups = np.asarray(mv_data['group_label'][0, 0, 0, 0, :])
        num_groups  = int(np.max(groups)+1)

    # analysis and plot functions ..............................................

    def write_tipping_points(data):
        mean = np.mean(data)
        path='~/utopia_output/n_{}.txt'.format(num_groups)
        file=open(os.path.expanduser(path), "a+")
        file.write("{}\n".format(mean))
        file.close()
        np.savetxt(hlpr.out_path.replace('sweep.pdf', 'tipping_points.csv'), data)

    def plot_pcrit_dist(data, cmap):
        """Plots the histogram of tipping points for the 2d tipping points plot.
        """
        hlpr.select_axis(0, 0)
        data=data.flatten()
        data=np.ma.masked_where(data<0, data)
        n_bins = len(dataset['homophily_parameter'])
        mean = np.mean(data)
        R_p = R_p_func(num_groups, mean)
        n, bins, patches = hlpr.ax.hist(data, n_bins, range=(0, 1))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        colormap = plt.cm.get_cmap(cmap)

        for c, p in zip(bin_centers, patches):
            plt.setp(p, 'facecolor', colormap(c))
        hlpr.ax.text(0.1, 0.95, r'$\overline{p_\mathrm{crit}}=$'+'{:.3f}'.format(mean, 3),
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=hlpr.ax.transAxes)
        hlpr.ax.set_xlim(0, 1)
        hlpr.ax.set_xlabel(r'$p_\mathrm{hom}$')
        hlpr.ax.set_ylabel(r'counts')
        hlpr.select_axis(0, 1)

    def plot_data_1d(param_plot, data_plot, std, plot_kwargs):
        if plot_prop == 'peaks_by_group':
            data_by_groups = [[data_plot[i][j]/bin_number for i in range(len(data_plot))] for j in range(num_groups)]
            for group in range(num_groups):
                recursive_update(plot_kwargs, {'label': "Group {}" "".format(group+1)})
                plot_kwargs['color']=colors[group]
                hlpr.ax.plot(param_plot, data_by_groups[group], **plot_kwargs)
        else:
            global color_idx
            plot_kwargs['color']=colors[color_idx%len(colors)]
            color_idx+=1
            if no_errors:
                hlpr.ax.plot(param_plot, data_plot, **plot_kwargs)
            else:
                hlpr.ax.errorbar(param_plot, data_plot, yerr=std, **plot_kwargs)

    def plot_data_2d(data_plot, param1, param2, plot_kwargs):
        if (plot_prop == 'convergence_time'):
            plot_kwargs['vmin']=0
            plot_kwargs['vmax']=dm['multiverse'].pspace.default["num_steps"]

        if (plot_prop=='tipping_points'):
            figure=plt.figure(figsize=[17, 6])
            hlpr.attach_figure_and_axes(fig=figure, axes=[[figure.add_subplot(121)], [figure.add_subplot(122)]])
            hlpr.select_axis(0, 1)
            heatmap = hlpr.ax.imshow(data_plot, aspect='auto', **plot_kwargs) #'RdYlGn_r'

        else:
            heatmap = hlpr.ax.imshow(data_plot, **plot_kwargs) #'RdYlGn_r'

        if len(param1) <= 10:
            hlpr.ax.set_xticks(np.arange(len(param1)))
            xticklabels = np.array([r'${:.2f}$'.format(p) for p in param1])
        else:
            hlpr.ax.set_xticks(np.arange(len(param1))
                            [::(int)(np.ceil(len(param1)/10))])
            xticklabels = np.array([r'${:.2f}$'.format(p)
                            for p in param1[::(int)(np.ceil(len(param1)/10))]])
        if len(param2) <= 10:
            hlpr.ax.set_yticks(np.arange(len(param2)))
            yticklabels = np.array([r'${:.2f}$'.format(p) for p in param2])
        else:
            hlpr.ax.set_yticks(np.arange(len(param2))
                            [::(int)(np.ceil(len(param2)/10))])
            yticklabels = np.array([r'${:.2f}$'.format(p)
                            for p in param2[::(int)(np.ceil(len(param2)/10))]])

        hlpr.ax.set_xticklabels(xticklabels)
        hlpr.ax.set_yticklabels(yticklabels)

        cbar = plt.colorbar(heatmap)
        if('homophily_parameter' in mv_data.coords):
            cbar.set_label(r'$p_\mathrm{hom}$')
        if(plot_prop == 'convergence_time'):
            cbar.set_label(r'$t_\mathrm{conv}$')


    def get_number_of_peaks(raw_data):
        final_state = raw_data.isel(time=-1)
        final_state = final_state[~np.isnan(final_state)]

        # binning of the final opinion distribution with binsize=1/bin_number
        hist_data,bin_edges = np.histogram(final_state, range=(0.,1.),
                                            bins=bin_number)

        peak_number = len(find_peaks(hist_data, prominence=15, distance=7)[0])

        return peak_number

    def get_localization(raw_data):
        final_state = raw_data.isel(time=-1)
        final_state = final_state[~np.isnan(final_state)]

        hist, bins = np.histogram(final_state, range=(0.,1.), bins=bin_number,
                                    density=True)
        hist *= 1/bin_number

        l = 0
        norm = 0
        for i in range(len(hist)):
            norm += hist[i]**4
            l += hist[i]**2

        l = norm/l**2

        return l

    def get_max_distance(raw_data):
        final_state = raw_data.isel(time=-1)
        final_state = final_state[~np.isnan(final_state)]
        min = 1.
        max = 0.
        for val in final_state:
            if val > max:
                max = val
            elif val < min:
                min = val

        return max-min

    def get_polarization(raw_data):
        final_state = raw_data.isel(time=-1)
        final_state = final_state[~np.isnan(final_state)]

        p = 0
        for i in range(len(final_state)):
            for j in range(len(final_state)):
                p += (final_state[i] - final_state[j])**2

        return p

    def get_final_variance(raw_data):
        final_state = raw_data.isel(time=-1)
        final_state = final_state[~np.isnan(final_state)]

        var = np.var(final_state.values)

        return var

    def get_convergence_time(raw_data):

        ct = np.inf
        num_users = len(raw_data.sel(time=0))
        for t in raw_data.coords['time']:
            data = raw_data.sel(time=t)
            hist, bins = np.histogram(data, range=(0, 1))
            bin_max = np.where(hist == hist.max())
            if(len(bin_max[0])>1):
                continue
            elif(hist[bin_max]>=0.9*num_users):
                ct = t
                break
        return ct

    def peaks_by_group(raw_data):
        num_users = len(raw_data.sel(time=0))
        #get the opinion arrays by group
        opinion = raw_data.isel(time=-1)
        opinion_by_groups = [opinion for i in range(num_groups)]
        for i in reversed(range(len(opinion))):
            for j in range(num_groups):
                if j!=groups[i]:
                    opinion_by_groups[j]=np.delete(opinion_by_groups[j], i, axis=0)

        #Find the peaks in each group.
        #If multiple peaks exist, average if sufficiently close, else discard
        peaks = []
        for j in range(num_groups):
            hist, bins = np.histogram(opinion_by_groups[j], range=(0., 1.))
            bin_max = np.where(hist == hist.max())

            if(len(bin_max[0])>1):
                if(bin_max[0][-1]-bin_max[0][0]<0.15*bin_number):
                    bin_max = np.average(bin_max)
                else:
                    #discard
                    bin_max = -1
            peaks=np.append(peaks, bin_max)
        return peaks

    def get_property(data, plot_prop: str=None):
        if plot_prop == 'number_of_peaks':
            return get_number_of_peaks(data)
        elif plot_prop == 'localization':
            return get_localization(data)
        elif plot_prop == 'max_distance':
            return get_max_distance(data)
        elif plot_prop == 'polarization':
            return get_polarization(data)
        elif plot_prop == 'final_variance':
            return get_final_variance(data)
        elif plot_prop == 'convergence_time':
            return get_convergence_time(data)
        elif plot_prop == 'peaks_by_group':
            return peaks_by_group(data)
        elif plot_prop == 'tipping_points':
            return get_convergence_time(data)
        else:
            raise ValueError("'plot_prop' invalid! Was: {}".format(plot_prop))

    # data handling and plot setup .............................................
    legend = False
    heatmap = False
    if not data_name in mv_data.data_vars:
        raise ValueError("'{}' not available in multiverse data."
                         " Available in multiverse field: {}"
                         "".format(data_name, mv_data.data_vars))

    # this is the dataset containing the chosen data to plot
    # for all parameter combinations
    dataset = mv_data[data_name]

    # number of different parameter values i.e. number of points in the graph
    num_param = len(dataset[dim])

    # initialize arrays containing the data to plot:
    if plot_prop == 'peaks_by_group':
        data_plot = [np.zeros(num_groups) for i in range(num_param)]
        param_plot = data_plot.copy()
        std = data_plot.copy()
    else:
        data_plot = np.zeros(num_param)
        param_plot = np.zeros(num_param)
        std = np.zeros_like(data_plot)

    # Get additional information for plotting
    # Get the title for the legend; not required if the second dimension has
    # length 1, eg. is a subspace plot for a single value
    if len(dataset[dim2])==1:
        leg_title=''
    else:
        leg_title = dim2
        if leg_title == "num_vertices":
            leg_title = r"$N$"
        elif leg_title =="weighting":
            leg_title = r"$\kappa$"
        elif leg_title == "rewiring":
            leg_title = r"$r$"
        elif leg_title == "p_rewire":
            leg_title = r"$p_{rewire}$"
        elif leg_title =="mean_degree":
            leg_title = r"$\bar{k}$"

    cmap_kwargs = plot_kwargs.pop("cmap_kwargs", None)
    if cmap_kwargs:
        cmin = cmap_kwargs.get("min", 0.)
        cmax = cmap_kwargs.get("max", 1.)
        cmap = cmap_kwargs.get("cmap")
        cmap = cm.get_cmap(cmap)
    markers = plot_kwargs.pop("markers", None)

    # If only one parameter sweep (dim) is done, the calculated quantity
    # is plotted against the parameter value.
    if (len(mv_data.coords) == 3):
        plot_kwargs = recursive_update(plot_kwargs_default_1d, plot_kwargs)
        param_index = 0
        for data in dataset:
            data_plot[param_index] = get_property(data, plot_prop)
            param_plot[param_index] = data[dim]
            param_index += 1
        if markers:
            plot_kwargs['marker'] = markers[0]
        plot_kwargs['color']=colors[color_idx%len(colors)]
        color_idx+=1
        plot_data_1d(param_plot, data_plot, std, plot_kwargs)

    # if two sweeps are done, check if the seed or the homophily param is sweeped
    elif (len(mv_data.coords) == 4):

        # average over the seed in this case
        if 'seed' in mv_data.coords:

            plot_kwargs = recursive_update(plot_kwargs_default_1d, plot_kwargs)

            for i in range(len(dataset[dim])):

                num_seeds = len(dataset['seed'])
                arr = np.zeros(num_seeds)

                for j in range(num_seeds):

                    data = dataset[{dim: i, 'seed': j}]
                    arr[j] = get_property(data, plot_prop)

                data_plot[i] = np.mean(arr)
                param_plot[i] = dataset[dim][i]
                std[i] = np.std(arr)

            if markers:
                plot_kwargs['marker'] = markers[0]
            plot_kwargs['color']=colors[color_idx]
            color_idx+=1
            plot_data_1d(param_plot, data_plot, std, plot_kwargs)

        # If 'stack', plot data of both dimensions against dim values (1d),
        # otherwise map data on 2d sweep parameter grid (color-coded).
        elif stack:
            legend = True
            plot_kwargs = recursive_update(plot_kwargs_default_1d, plot_kwargs)
            param_plot = dataset[dim]
            for i in range(len(dataset[dim2])):
                for j in range(num_param):
                    data = dataset[{dim2: i, dim: j}]
                    data_plot[j] = get_property(data, plot_prop)
                    recursive_update(plot_kwargs, {'label': "{}"
                                        "".format(dataset[dim2][i].data)})
                if cmap_kwargs:
                    c = i * (cmax - cmin) / (len(dataset[dim2])-1.) + cmin
                    recursive_update(plot_kwargs, {'color': cmap(c)})
                if markers:
                    plot_kwargs['marker'] = markers[i%len(markers)]
                plot_data_1d(param_plot, data_plot, std, plot_kwargs)

        else:

            plot_kwargs = recursive_update(plot_kwargs_default_2d, plot_kwargs)
            heatmap = True
            num_param2 = len(dataset[dim2])
            data_plot = np.zeros((num_param2, num_param))
            param1 = np.zeros(num_param)
            param2 = np.zeros(num_param2)

            for i in range(num_param):
                param1[i] = dataset[dim][i]

                for j in range(num_param2):
                    param2[j] = dataset[dim2][j]
                    data = dataset[{dim: i, dim2: j}]
                    data_plot[j,i] = get_property(data, plot_prop)

            plot_data_2d(data_plot, param1, param2, plot_kwargs)

    elif (len(mv_data.coords) == 5):
        if('seed' in mv_data.coords):
            num_seeds = len(dataset['seed'])

            if stack:

                legend = True
                plot_kwargs = recursive_update(plot_kwargs_default_1d, plot_kwargs)
                param_plot = dataset[dim]
                param2 = dataset[dim2]

                for i in range(len(param2)):

                    for j in range(num_param):

                        arr = np.zeros(num_seeds)

                        for k in range(num_seeds):
                            data = dataset[{dim2: i, dim: j, 'seed': k}]
                            arr[k] = get_property(data, plot_prop)

                        data_plot[j] = np.mean(arr)

                    recursive_update(plot_kwargs, {'label': "{}"
                                        "".format(dataset[dim2][i].data)})
                    if cmap_kwargs:
                        c = i * (cmax - cmin) / (len(dataset[dim2])-1.) + cmin
                        recursive_update(plot_kwargs, {'color': cmap(c)})

                    if markers:
                        plot_kwargs['marker'] = markers[i%len(markers)]

                    plot_data_1d(param_plot, data_plot, std, plot_kwargs)

            else:

                plot_kwargs = recursive_update(plot_kwargs_default_2d, plot_kwargs)
                heatmap = True
                num_param2 = len(dataset[dim2])
                data_plot = np.zeros((num_param2, num_param))
                param1 = np.zeros(num_param)
                param2 = np.zeros(num_param2)

                for i in range(num_param):
                    param1[i] = dataset[dim][i]

                    for j in range(num_param2):
                        param2[j] = dataset[dim2][j]
                        arr = np.zeros(num_seeds)

                        for k in range(num_seeds):
                            data = dataset[{dim: i, dim2: j, 'seed': k}]
                            arr[k] = get_property(data, plot_prop)

                        data_plot[j,i] = np.mean(arr)

                plot_data_2d(data_plot, param1, param2, plot_kwargs)

        elif('homophily_parameter' in mv_data.coords):
            if stack:
                raise TypeError("Cannot stack 2d homophily params plot")
            else:
                plot_kwargs = recursive_update(plot_kwargs_default_2d, plot_kwargs)
                #exclude 'bad' values
                plot_kwargs['vmin']=0
                plot_kwargs['vmax']=1
                heatmap = True
                num_param2 = len(dataset[dim2])
                num_param3 = len(dataset['homophily_parameter'])
                data_plot = -1*np.ones((num_param2, num_param))
                l = num_param*num_param2*num_param3
                param1 = np.asarray([dataset[dim][n].data for n in range(num_param)])
                param2 = np.asarray([dataset[dim2][m].data for m in range(num_param2)])
                for i in range(num_param):
                    if param1[i]==0:
                        continue
                    for j in range(num_param2):
                        if param2[j]==0:
                            continue
                        #check the model converges at p=0:
                        if(get_convergence_time(dataset[{dim: i, dim2: j, 'homophily_parameter': 0}])==np.inf):
                            continue
                        for k in reversed(range(num_param3)):
                            data = dataset[{dim: i, dim2: j, 'homophily_parameter': k}]
                            if get_convergence_time(data)==np.inf:
                                continue
                            else:
                                data_plot[j, i]=dataset['homophily_parameter'][k]
                                break
                #exclude 'bad' values
                data_plot=np.ma.masked_where(data_plot<0, data_plot)
                write_tipping_points(data_plot.flatten())
                if not write_only:
                    plot_data_2d(data_plot, param1, param2, plot_kwargs)
                    plot_pcrit_dist(data_plot, plot_kwargs['cmap'])

    # Add formatted labels and title
    def label(input):
        if(input == "tolerance"):
            return r'tolerance $\epsilon$'
        elif(input == "susceptibility"):
            return r'susceptibility $\mu$'
        elif(input == "homophily_parameter"):
            return r'homophily parameter $p_\mathrm{hom}$'
        elif (input == 'mean_degree'):
            return r'mean degree $\kappa$'
        elif input == "localization":
            return r'$L$'
        elif plot_prop == "final_variance":
            return r'var($\sigma$)'
        elif plot_prop == 'number_of_peaks':
            return r'$N_\mathrm{peaks}$'
        elif plot_prop == 'max_distance':
            return '$d_{max}$'
        elif plot_prop == 'convergence_time':
            return r'$t_\mathrm{conv}$'
        elif plot_prop == 'peaks_by_group':
            return r'peak position'
        else:
            return input

    def title(input):
        if input == "localization":
            return r'$\textbf{Localisation}'
        elif input == 'number_of_peaks':
            return r'$\textbf{Number of peaks by group}$'
        elif input == 'convergence_time':
            return r'$\textbf{Convergence time}$'
        elif input == 'peaks_by_group':
            return r'$\textbf{Peaks by group}$'
        elif input == 'tipping_points':
            return r'$\textbf{Tipping points}$'
        else:
            return input

    if stack:
        if(dim=='homophily_parameter'):
            hlpr.ax.set_xlim(0., 1.)
        hlpr.ax.grid(b=True, linewidth =0.5)
        hlpr.provide_defaults('set_labels', **{'x': label(dim), 'y': label(plot_prop)})
        hlpr.provide_defaults('set_title', **{'title': title(plot_prop)})
    else:
        hlpr.provide_defaults('set_labels', **{'x': label(dim), 'y': label(dim2)})
        hlpr.provide_defaults('set_title', **{'title': ''}) #title(plot_prop)

    if legend:
        hlpr.ax.legend(title=leg_title, loc='best')

    # Set minor ticks
    if plot_prop == "max_distance" or plot_prop == "localization":
        hlpr.ax.get_xaxis().set_major_locator(ticker.MultipleLocator(0.1))
        hlpr.ax.get_xaxis().set_minor_locator(ticker.MultipleLocator(0.05))
        hlpr.ax.get_yaxis().set_minor_locator(ticker.MultipleLocator(0.1))
