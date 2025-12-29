'''Give this python script the root directory of experiments and it will generate plots
for each lower parameter sweep inside of that directory
'''

from pathlib import Path
from influence.plotting import plot_comparison_tree, DEFAULT_FITNESS_NAME, LEGEND_LOC_CHOICES
from influence.parsing import BatchLinePlotParser

if __name__ == '__main__':
    parser = BatchLinePlotParser(
        prog='comparison_tree.py',
        description='plot the experiments in the specified directory',
        epilog=''
    )
    parser.add_argument(
        'root_dir',
        help='root directory with all the experiments',
        type=str
    )
    parser.add_argument(
        'out_dir',
        help='directory to save plots to',
        type=str,
        nargs='?',
        default=None
    )
    parser.add_argument(
        '--fitness-colors',
        help='plot with consistent colors based on names of fitness shaping methods',
        action='store_true'
    )
    parser.add_argument(
        '--legend-order',
        type=str,
        choices=['acm-telo'],
        default=None,
        help='order the legend (default: no reordering)'
    )
    parser.add_argument(
        '--legend-loc',
        type=str,
        choices=LEGEND_LOC_CHOICES,
        default='best',
        help='specify location of the legend (default: best)'
    )
    parser.add_argument(
        '--csv-name',
        help='name of csv to use for fitness',
        default=DEFAULT_FITNESS_NAME
    )
    parser.add_plot_args()

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    plot_comparison_tree(
        Path(args.root_dir),
        out_dir,
        args.fitness_colors,
        args.legend_order,
        args.legend_loc,
        args.csv_name,
        parser.dump_batch_plot_args(args),
        parser.dump_batch_line_plot_args(args)
    )
