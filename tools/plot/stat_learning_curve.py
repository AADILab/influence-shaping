'''Give this python script the parent directoy of trials and it will plot the statistics of these trials'''

from pathlib import Path
from influence.plotting import plot_stat_learning_curve, DEFAULT_FITNESS_NAME
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
    parser.add_argument(
        '--individual_trials',
        help='plot each trial as a different color',
        action='store_true'
    )
    parser.add_argument(
        '--csv_name',
        help='name of csv to use for fitness',
        type=str,
        default=DEFAULT_FITNESS_NAME
    )
    args = parser.parse_args()

    plot_stat_learning_curve(Path(args.trials_dir), args.individual_trials, args.csv_name, parser.dump_line_plot_args(args), parser.dump_plot_args(args))
