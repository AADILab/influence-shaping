'''Give this python script the root directory of experiments and it will
generate a learning curve plot for each config based on specified commandline arguments
'''

from pathlib import Path
from influence.plotting import plot_learning_curve_tree
from influence.parsing import BatchLinePlotParser

if __name__=='__main__':
    parser = BatchLinePlotParser(
        prog='learning_curve_tree.py',
        description='plot the learning curves of all fitness.csv files in directory tree',
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
        '--individual_agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    parser.add_plot_args()

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    plot_learning_curve_tree(Path(args.root_dir), out_dir, args.individual_agents, parser.dump_batch_plot_args(args), parser.dump_batch_line_plot_args(args))
