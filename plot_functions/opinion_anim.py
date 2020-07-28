import logging
import numpy as np
import matplotlib.pyplot as plt

from utopya import DataManager, UniverseGroup
from utopya.plotting import UniversePlotCreator, PlotHelper, is_plot_func

from .tools import setup_figure

log = logging.getLogger(__name__)
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

#-------------------------------------------------------------------------------
@is_plot_func(creator_type=UniversePlotCreator,
              supports_animation=True, helper_defaults=dict(
              set_labels=dict(x="Values", y="Counts")))
def opinion_animation(dm: DataManager, *,
                      uni: UniverseGroup,
                      hlpr: PlotHelper,
                      num_bins: int=100,
                      time_idx: int,
                      title: str=None,
                      val_range: tuple=(0., 1.)):

    """Plots an animated histogram of the opinion distribution over time. If
    the model mode is 'conflict_undir', the opinion distribution of the
    discriminators and non-discriminators is also shown.

    Arguments:
        num_bins(int): Binning of the histogram
        time_idx (int, optional): Only plot one single frame (eg. last frame)
        title (str, optional): Custom plot title
        val_range(int, optional): Value range of the histogram
    """

    #figure layout..............................................................
    #the 'conflict_undir' has a non-standard plot layout with two additional axis
    #for the discriminators' and non-discriminators' opinion distributions
    if uni['cfg']['OpDisc']['mode']=='conflict_undir':
        disc_plot = True
        figure, axs = setup_figure(uni['cfg'], plot_name='opinion_anim',
            title=title, figsize=(10, 15), ncols=2, nrows=3,
            height_ratios=[1, 6, 3], width_ratios=[1, 1],
            gridspec=[(0, slice(0, 2)), (1, slice(0, 2)), (2, 0), (2, 1)])
    else:
        disc_plot = False
        figure, axs = setup_figure(uni['cfg'], plot_name='opinion_anim', title=title)
    hlpr.attach_figure_and_axes(fig=figure, axes=axs)

    #datasets...................................................................
    opinions    = uni['data/OpDisc/nw/opinion']
    time        = opinions['time'].data
    time_steps  = time.size
    #dict containing the data to plot, as well axis-specific info
    to_plot = {'all': {'data': opinions, 'axs_idx': 1, 'text': '',
                           'color': 'dodgerblue'}}

    #data analysis..............................................................
    if disc_plot:
        #calculate the opinions of only the discriminators and non-discriminators
        #respectively and add to the dict
        discriminators = uni['data/OpDisc/nw/discriminators']
        p_disc = uni['cfg']['OpDisc']['discriminators']
        mask = np.empty(opinions.shape, dtype=bool)
        mask[:,:] = (discriminators == 0)
        ops_disc = np.ma.MaskedArray(opinions, mask)
        ops_nondisc = np.ma.MaskedArray(opinions, ~mask)

        to_plot['disc'] = {'data': ops_disc, 'axs_idx': 2,
            'color': 'teal', 'text': f'discriminators ($p_d$={p_disc})'}

        to_plot['nondisc'] = {'data': ops_nondisc, 'axs_idx': 3,
            'color': 'mediumaquamarine',
            'text': f'discriminators ($1-p_d$={1-p_disc})'}

    #get histograms.............................................................
    def get_hist_data(input_data):
        counts, bin_edges = np.histogram(input_data, range=val_range, bins=num_bins)
        bin_pos = bin_edges[:-1] + (np.diff(bin_edges) / 2.)

        return counts, bin_edges, bin_pos

    bars = {}
    t = time_idx if time_idx else range(time_steps)
    #calculate histograms, set axis ranges, set axis descriptions in upper left
    #corners
    for key in to_plot.keys():
        counts, bin_edges, pos = get_hist_data(to_plot[key]['data'][t, :])
        hlpr.select_axis(0, to_plot[key]['axs_idx'])
        hlpr.ax.set_xlim(val_range)
        bars[key] = hlpr.ax.bar(pos, counts, width=np.diff(bin_edges),
                                color=to_plot[key]['color'])

    for key in to_plot.keys():
        hlpr.select_axis(0, to_plot[key]['axs_idx'])
        to_plot[key]['text']=hlpr.ax.text(0.02, 0.93, to_plot[key]['text'],
                                          transform=hlpr.ax.transAxes)

    #animate....................................................................
    def update_data(stepsize: int=1):
        """Updates the data of the imshow objects"""
        if time_idx:
            log.info(f"Plotting distribution at time step {time[time_idx]} ...")
        else:
            log.info(f"Plotting animation with {opinions.shape[0] // stepsize} "
                      "frames ...")
        next_frame_idx = 0
        if time_steps < stepsize:
            log.warn("Stepsize is greater than number of steps. Continue by "
                     "plotting fist and last frame.")
            stepsize=time_steps-1
        for t in range(time_steps):
            if t < next_frame_idx:
                continue
            if time_idx:
                t = time_idx
            for key in to_plot.keys():
                hlpr.select_axis(0, to_plot[key]['axs_idx'])
                data = to_plot[key]['data'][t, :]
                if key != 'all':
                    #for disc/non-disc plots, the data is a masked array and needs
                    #to be compressed (removing None values)
                    data = data.compressed()
                counts_at_t, _, _ = get_hist_data(data)
                for idx, rect in enumerate(bars[key]):
                    rect.set_height(counts_at_t[idx])
                if key == 'all':
                    to_plot[key]['text'].set_text(f'step {time[t]}')
                    hlpr.ax.relim()
                    hlpr.ax.autoscale_view(scalex=False)
                    y_max = hlpr.ax.get_ylim()
                else:
                    #rescale ylim to same value for all plots
                    hlpr.ax.set_ylim(y_max)
            if time_idx:
                yield
                break
            next_frame_idx = t + stepsize
            yield

    hlpr.register_animation_update(update_data)
