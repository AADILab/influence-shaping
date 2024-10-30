'''Give this python script an experiment directory (parent of multiple parents of trial directories) 
and it will plot the statistics of each parameter combination against each other
'''

import argparse
from pathlib import Path
from influence.plotting import plot_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='experiment.py',
        description='plot the experiment from the specified directory',
        epilog=''
    )
    parser.add_argument(
        'experiment_dir',
        help='parent directory of directories with trials in them',
        type=str
    )
    parser.add_argument(
        '-s', '--silent',
        help='run silently, without showing the plot',
        action='store_true',
    )
    parser.add_argument(
        '-o', '--output', 
        help='directory to output image of plot to',
        type=str
    )
    parser.add_argument(
        '--title',
        help='title of generated plot',
        type=str
    )
    parser.add_argument(
        '--individual_agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    args = parser.parse_args()

    if args.output is not None:
        args.output = Path(args.output)

    plot_experiment(Path(args.experiment_dir), args.output, args.silent, args.title)
