'''Give this python script the root directory of experiments and it will
generate a plot for each config based on specified commandline arguments
'''

from pathlib import Path
from influence.plotting import plot_stat_learning_curve_tree, DEFAULT_FITNESS_NAME
from influence.parsing import BatchLinePlotParser

if __name__=='__main__':
    parser = BatchLinePlotParser(
        prog='stat_learning_curve_tree.py',
        description='plot the stats of all configs in directory',
        epilog=''
    )
    parser.add_argument(
        'root_dir',
        help='root directory with all the experiments'
    )
    parser.add_argument(
        'out_dir',
        help='directory to save plots to',
        type=str,
        nargs='?',
        default=None
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
    parser.add_plot_args()

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    plot_stat_learning_curve_tree(Path(args.root_dir), out_dir, args.individual_trials, args.csv_name, parser.dump_batch_plot_args(args), parser.dump_batch_line_plot_args(args))
