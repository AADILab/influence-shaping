'''Give this python script a comparison directory (parent of multiple parents of trial directories)
and it will plot the statistics of each parameter combination against each other
'''

from pathlib import Path
from influence.plotting import plot_comparison, DEFAULT_FITNESS_NAME
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='comparison.py',
        description='plot the experiment from the specified directory',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'comparison_dir',
        help='parent directory of directories with trials in them',
        type=str
    )
    parser.add_argument(
        '--fitness_colors',
        help='use defined colors for plotting results of fitness shaping methods',
        action='store_true'
    )
    parser.add_argument(
        '--csv_name',
        help='name of csv to use for fitness',
        type=str,
        default=DEFAULT_FITNESS_NAME
    )
    args = parser.parse_args()

    plot_comparison(Path(args.comparison_dir), args.fitness_colors, args.csv_name, parser.dump_line_plot_args(args), parser.dump_plot_args(args))
