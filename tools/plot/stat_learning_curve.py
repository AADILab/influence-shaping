'''Give this python script the parent directoy of trials and it will plot the statistics of these trials'''

from pathlib import Path
from influence.plotting import plot_stat_learning_curve
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='stat_learning_curve.py',
        description='plot the statistics from the trials in the specified directory',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'trials_dir',
        help='parent directory of trials',
        type=str
    )
    args = parser.parse_args()

    plot_stat_learning_curve(Path(args.trials_dir), parser.dump_line_plot_args(args), parser.dump_plot_args(args))
