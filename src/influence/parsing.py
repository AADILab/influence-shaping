import argparse
from pathlib import Path
import os
from typing import List, Optional
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def moving_average_filter(arr, window_size: int):
    result = np.copy(arr).astype(float)
    half_window = window_size // 2

    for i in range(len(arr)):
        # Try to center the window around point i
        start_idx = max(0, i - half_window)
        end_idx = min(len(arr), i + half_window + 1)

        # If we can't get the full window, use what's available
        result[i] = np.mean(arr[start_idx:end_idx])

    return result

class PlotArgs():
    def __init__(self,
            title: Optional[str]=None,
            output: Optional[str]=None,
            xlim: Optional[List[float]]=None,
            ylim: Optional[List[float]]=None,
            xticks: Optional[List[float]]=None,
            yticks: Optional[List[float]]=None,
            xlabel: Optional[str]=None,
            ylabel: Optional[str]=None,
            figsize: Optional[List[float]]=None,
            axes_position: Optional[List[float]]=None,
            silent: bool=False,
            remove_border: bool=False,
            dpi: Optional[int]=None,
            legend_facecolor: Optional[str]=None
        ):
        self.title = title
        if output is not None:
            self.output = Path(output)
        else:
            self.output = output
        self.xlim = xlim
        self.ylim = ylim
        self.xticks = xticks
        self.yticks = yticks
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.axes_position = axes_position
        self.silent = silent
        self.remove_border = remove_border
        self.dpi = dpi
        self.legend_facecolor = legend_facecolor

    def init_figure(self, nrows:int=1, ncols:int=1):
        return plt.subplots(nrows, ncols, figsize=self.figsize)

    def apply(self, ax: Axes):
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.yticks is not None:
            ax.set_yticks(self.yticks)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.axes_position is not None:
            ax.set_position(self.axes_position)
        if self.remove_border:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        if self.legend_facecolor is not None:
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor(self.legend_facecolor)

    def finish_figure(self, fig: Figure):
        if self.output is not None:
            if not os.path.exists(self.output.parent):
                os.makedirs(self.output.parent)
            fig.savefig(self.output, dpi=self.dpi)
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
            xlim: Optional[List[float]]=None,
            ylim: Optional[List[float]]=None,
            xticks: Optional[List[float]]=None,
            yticks: Optional[List[float]]=None,
            xlabel: Optional[str]=None,
            ylabel: Optional[str]=None,
            silent: bool=False,
            remove_border: bool=False,
            dpi: Optional[int]=None
        ):
        self.xlim = xlim
        self.ylim = ylim
        self.xticks = xticks
        self.yticks = yticks
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.silent = silent
        self.remove_border = remove_border
        self.dpi = dpi

    def build_plot_args(self, title: Optional[str], output: Optional[str]):
        return PlotArgs(
            title=title,
            output=output,
            xlim=self.xlim,
            ylim=self.ylim,
            xticks=self.xticks,
            yticks=self.yticks,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            silent=self.silent,
            remove_border=self.remove_border,
            dpi=self.dpi
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
            '--xticks',
            nargs='*',
            type=float,
            help='Custom x-tick positions (space separated). Use empty to remove all x-ticks.'
        )
        self.add_argument(
            '--yticks',
            nargs='*',
            type=float,
            help='Custom y-tick positions (space separated). Use empty to remove all y-ticks.'
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
            '--figsize',
            nargs=2,
            type=float,
            metavar=('WIDTH', 'HEIGHT'),
            help='Figure size in inches: width height'
        )
        self.add_argument(
            '--axes-position',
            nargs=4,
            type=float,
            metavar=('LEFT', 'BOTTOM', 'WIDTH', 'HEIGHT'),
            help='Axes position as fractions of figure: left bottom width height (all between 0 and 1)'
        )
        self.add_argument(
            '-s', '--silent',
            help='run silently without showing the plot',
            action='store_true'
        )
        self.add_argument(
            '--remove-border',
            help='remove border of the figure',
            action='store_true'
        )
        self.add_argument(
            '--dpi',
            help='DPI (dots per inch) for saved plots',
            type=int,
            default=None
        )
        self.add_argument(
            '--legend-facecolor',
            help='background color of the legend (any matplotlib color string, e.g. "lightgray", "0.7")',
            type=str,
            default=None
        )
        return None

    def dump_plot_args(self, args):
        return PlotArgs(
            title=args.title,
            output=args.output,
            xlim=args.xlim,
            ylim=args.ylim,
            xticks=args.xticks,
            yticks=args.yticks,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=args.figsize,
            axes_position=args.axes_position,
            silent=args.silent,
            remove_border=args.remove_border,
            dpi=args.dpi,
            legend_facecolor=args.legend_facecolor
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
        return LinePlotArgs(
            window_size=args.window_size,
            downsample=args.downsample
        )

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
            '--xticks',
            nargs='*',
            type=float,
            help='Custom x-tick positions (space separated). Use empty to remove all x-ticks.'
        )
        self.add_argument(
            '--yticks',
            nargs='*',
            type=float,
            help='Custom y-tick positions (space separated). Use empty to remove all y-ticks.'
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
            '--figsize',
            nargs=2,
            type=float,
            metavar=('WIDTH', 'HEIGHT'),
            help='Figure size in inches: width height'
        )
        self.add_argument(
            '--axes-position',
            nargs=4,
            type=float,
            metavar=('LEFT', 'BOTTOM', 'WIDTH', 'HEIGHT'),
            help='Axes position as fractions of figure: left bottom width height (all between 0 and 1)'
        )
        self.add_argument(
            '-s', '--silent',
            help='run silently without showing any plots',
            action='store_true'
        )
        self.add_argument(
            '--remove-border',
            help='remove border of the figure',
            action='store_true'
        )
        self.add_argument(
            '--dpi',
            help='DPI (dots per inch) for saved plots',
            type=int,
            default=None
        )
        return None

    def dump_batch_plot_args(self, args):
        return BatchPlotArgs(
            xlim=args.xlim,
            ylim=args.ylim,
            xticks=args.xticks,
            yticks=args.yticks,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            silent=args.silent,
            remove_border=args.remove_border,
            dpi=args.dpi
        )

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
