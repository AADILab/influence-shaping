'''Give this python script a comparison directory (parent of multiple parents of trial directories)
and it will plot the statistics of each parameter combination against each other
'''

from pathlib import Path
from influence.plotting import plot_comparison, DEFAULT_FITNESS_NAME, LEGEND_LOC_CHOICES
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
        '--fitness-colors',
        help='use defined colors for plotting results of fitness shaping methods',
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
        '--no-legend',
        action='store_true',
        help='turn off the legend'
    )
    parser.add_argument(
        '--csv-name',
        help='name of csv to use for fitness',
        type=str,
        default=DEFAULT_FITNESS_NAME
    )
    args = parser.parse_args()

    plot_comparison(
        experiment_dir=Path(args.comparison_dir),
        use_fitness_colors=args.fitness_colors,
        legend_order=args.legend_order,
        legend_loc=args.legend_loc,
        no_legend=args.no_legend,
        csv_name=args.csv_name,
        line_plot_args=parser.dump_line_plot_args(args),
        plot_args=parser.dump_plot_args(args)
    )
