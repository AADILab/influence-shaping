import argparse
from pathlib import Path
import os
from typing import List, Optional, Any
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def moving_average_filter(arr, window_size: int):
    pad_len = window_size - 1
    new_arr = np.concatenate([
        np.ones(pad_len)*arr[0],
        arr
    ])
    return np.convolve(
        new_arr,
        np.ones(window_size)/window_size,
        mode='valid'
    )

class PlotArgs():
    def __init__(self,
            title: Optional[str]=None,
            output: Optional[str]=None,
            xlim: Optional[List[float]]=None,
            ylim: Optional[List[float]]=None,
            xlabel: Optional[str]=None,
            ylabel: Optional[str]=None,
            titlesize: Optional[float]=None,
            titlepad: Optional[float]=None,
            xlabelsize: Optional[float]=None,
            ylabelsize: Optional[float]=None,
            xticklabelsize: Optional[float]=None,
            yticklabelsize: Optional[float]=None,
            legendtextsize: Optional[float]=None,
            legendloc: Optional[Any]=None,
            figsize: Optional[List[float]]=None,
            style: Optional[str]=None,
            silent: bool=False
        ):
        self.title = title
        if output is not None:
            self.output = Path(output)
        else:
            self.output = output
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.titlesize = titlesize
        self.titlepad = titlepad
        self.xlabelsize = xlabelsize
        self.ylabelsize = ylabelsize
        self.xticklabelsize = xticklabelsize
        self.yticklabelsize = yticklabelsize
        self.legendtextsize = legendtextsize
        self.legendloc = legendloc
        self.figsize = figsize
        self.style = style
        self.silent = silent

    def apply(self, ax: Axes):
        if self.title:
            pad = 0
            if self.titlepad:
                pad = self.titlepad
            ax.set_title(self.title, pad=pad)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.titlesize:
            ax.title.set_fontsize(self.titlesize)
        if self.xlabelsize:
            ax.xaxis.label.set_fontsize(self.xlabelsize)
        if self.ylabelsize:
            ax.yaxis.label.set_fontsize(self.ylabelsize)
        if self.xticklabelsize:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=self.xticklabelsize)
        if self.yticklabelsize:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=self.yticklabelsize)
        if self.legendtextsize and (legend:=ax.get_legend()):
            for text in legend.get_texts():
                text.set_fontsize(self.legendtextsize)
        if self.legendloc and(legend:=ax.get_legend()):
            legend.set_loc(self.legendloc)

        if self.style == 'gray':
            # Set the background color
            ax.set_facecolor((0.9, 0.9, 0.9, 1))  # Set background to grey

            # Customize the grid lines
            ax.set_axisbelow(True)
            ax.grid(color='white', linestyle='-', linewidth=1, zorder=0)  # Set grid lines to white

    def finish_figure(self, fig: Figure):
        if self.figsize is not None:
            fig.set_size_inches(self.figsize[0], self.figsize[1])
        fig.tight_layout()
        if self.output is not None:
            if not os.path.exists(self.output.parent):
                os.makedirs(self.output.parent)
            fig.savefig(self.output)
            plt.close(fig)

        if not self.silent:
            plt.show()

class LinePlotArgs():
    def __init__(self, window_size: Optional[int], downsample: int):
        self.window_size = window_size
        self.downsample = downsample

    def get_ys(self, ys):
        if self.window_size:
            return moving_average_filter(ys, self.window_size)
        else:
            return ys

    def get_pts(self, xs, ys):
        return xs[::self.downsample], self.get_ys(ys)[::self.downsample]

class BatchPlotArgs():
    def __init__(self,
            xlim: Optional[List[float]],
            ylim: Optional[List[float]],
            xlabel: Optional[str],
            ylabel: Optional[str],
            silent: bool
        ):
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.silent = silent

    def build_plot_args(self, title: Optional[str], output: Optional[str]):
        return PlotArgs(
            title=title,
            output=output,
            xlim=self.xlim,
            ylim=self.ylim,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            silent=self.silent
        )

class BatchLinePlotArgs():
    def __init__(self,
            window_size: Optional[int],
            downsample: int
        ):
        self.window_size = window_size
        self.downsample = downsample

    def build_line_plot_args(self):
        return LinePlotArgs(
            window_size=self.window_size,
            downsample=self.downsample
        )

class PlotParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_styles = ['gray']

    def add_plot_args(self):
        self.add_argument(
            '--title',
            help='title of generated plot',
            type=str
        )
        self.add_argument(
            '-o', '--output',
            help='directory to output image of plot to',
            type=str
        )
        self.add_argument(
            '--xlim',
            nargs=2,
            help='min and max xlimit for plot',
            type=float
        )
        self.add_argument(
            '--ylim',
            nargs=2,
            help='min and max ylimit for plot',
            type=float
        )
        self.add_argument(
            '--xlabel',
            help='label for the x axis',
            type=str
        )
        self.add_argument(
            '--ylabel',
            help='label for the y axis',
            type=str
        )
        self.add_argument(
            '--titlesize',
            help='size of title',
            type=float
        )
        self.add_argument(
            '--titlepad',
            help='pad to add spacing between title and plot',
            type=float
        )
        self.add_argument(
            '--xlabelsize',
            help='size of xlabel text',
            type=float
        )
        self.add_argument(
            '--ylabelsize',
            help='size of ylabel text',
            type=float
        )
        self.add_argument(
            '--xticklabelsize',
            help='size of xtick labels',
            type=float
        )
        self.add_argument(
            '--yticklabelsize',
            help='size of ytick labels',
            type=float
        )
        self.add_argument(
            '--legendtextsize',
            help='size of text in legend',
            type=float
        )
        self.add_argument(
            '--legendloc',
            help='location of legend',
            type=str
        )
        self.add_argument(
            '--figsize',
            nargs=2,
            help='size in inches of the figure',
            type=float
        )
        self.add_argument(
            '--style',
            type=str,
            choices=self.valid_styles,
            help=f"Choose one of the following to add a style to the plot: {', '.join(self.valid_styles)} (default: None)"
        )
        self.add_argument(
            '-s', '--silent',
            help='run silently without showing the plot',
            action='store_true'
        )
        return None

    def dump_plot_args(self, args):
        return PlotArgs(
            args.title,
            args.output,
            args.xlim,
            args.ylim,
            args.xlabel,
            args.ylabel,
            args.titlesize,
            args.titlepad,
            args.xlabelsize,
            args.ylabelsize,
            args.xticklabelsize,
            args.yticklabelsize,
            args.legendtextsize,
            args.legendloc,
            args.figsize,
            args.style,
            args.silent
        )

class LinePlotParser(PlotParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_plot_args(self):
        super().add_plot_args()
        self.add_argument(
            '--window_size',
            help='window size for moving average filter on final plot',
            type=int
        )
        self.add_argument(
            '--downsample',
            help='downsample and only plot one point every _ points',
            type=int,
            default=1
        )
        return None

    def dump_line_plot_args(self, args):
        return LinePlotArgs(args.window_size, args.downsample)

class BatchPlotParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_plot_args(self):
        self.add_argument(
            '--xlim',
            nargs=2,
            help='min and max xlimits for plots',
            type=float
        )
        self.add_argument(
            '--ylim',
            nargs=2,
            help='min and max ylimits for plots',
            type=float
        )
        self.add_argument(
            '--xlabel',
            help='label for x axes',
            type=str
        )
        self.add_argument(
            '--ylabel',
            help='label for the y axes',
            type=str
        )
        self.add_argument(
            '-s', '--silent',
            help='run silently without showing any plots',
            action='store_true'
        )
        return None

    def dump_batch_plot_args(self, args):
        return BatchPlotArgs(args.xlim, args.ylim, args.xlabel, args.ylabel, args.silent)

class BatchLinePlotParser(BatchPlotParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_plot_args(self):
        super().add_plot_args()
        self.add_argument(
            '--window_size',
            help='window size for moving average filter on plots',
            type=int
        )
        self.add_argument(
            '--downsample',
            help='downsample and only plot one point every _ points',
            type=int,
            default=1
        )
        return None

    def dump_batch_line_plot_args(self, args):
        return BatchLinePlotArgs(args.window_size, args.downsample)
