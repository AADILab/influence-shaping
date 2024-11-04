import argparse
from pathlib import Path
import os
from typing import List, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class PlotArgs():
    def __init__(self, title: Optional[str]=None, output: Optional[str]=None, 
            xlim: Optional[List[float]]=None, ylim: Optional[List[float]]=None, 
            xlabel: Optional[str]=None, ylabel: Optional[str]=None,
            silent: bool=False):
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
        
        if not self.silent:
            plt.show()

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
