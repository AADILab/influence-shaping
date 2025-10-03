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
            xlabel: Optional[str]=None, 
            ylabel: Optional[str]=None,
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
        self.silent = silent
    
    def apply(self, ax: Axes):
        if self.title:
            ax.set_title(self.title)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
    
    def finish_figure(self, fig: Figure):
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
            '-s', '--silent',
            help='run silently without showing the plot',
            action='store_true'
        )
        return None

    def dump_plot_args(self, args):
        return PlotArgs(args.title, args.output, args.xlim, args.ylim, args.xlabel, args.ylabel, args.silent)

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
