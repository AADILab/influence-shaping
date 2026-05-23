'''Give this python script a top-level directory containing multiple comparison directories
and it will plot a grouped bar chart where each group of bars represents one comparison.

Expected layout:
    parent_dir/
        comparison1/
            method_a/ trial_0/ ...
            method_b/ trial_0/ ...
        comparison2/
            ...
'''

from pathlib import Path
from influence.plotting import plot_multibar_comparison, DEFAULT_FITNESS_NAME, LABELMAP_CHOICES, LEGEND_LOC_CHOICES, BAR_ORDER_CHOICES
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='single_multibar_comparison.py',
        description='plot grouped bar chart of performance from multiple comparison directories',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'parent_dir',
        help='directory containing multiple comparison subdirectories',
        type=str
    )
    parser.add_argument(
        '--fitness-colors',
        help='use defined colors for plotting results of fitness shaping methods',
        action='store_true'
    )
    parser.add_argument(
        '--generation',
        help='generation index to snapshot (default: final generation)',
        type=int,
        default=None
    )
    parser.add_argument(
        '--showbest',
        help='draw a dashed line at the highest possible score',
        action='store_true'
    )
    parser.add_argument(
        '--normalize-yscores',
        help='divide each bar by its comparison\'s maximum achievable score for fair cross-comparison',
        action='store_true'
    )
    parser.add_argument(
        '--colorblind',
        help='add distinct hatch patterns to each method for colorblind-friendly plots',
        action='store_true'
    )
    parser.add_argument(
        '--no-legend',
        action='store_true',
        help='turn off the legend'
    )
    parser.add_argument(
        '--legend-order',
        type=str,
        choices=['acm-telo', 'jaamas'],
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
        '--bar-order',
        type=str,
        choices=BAR_ORDER_CHOICES,
        default=None,
        help='order bars within each group left-to-right (default: no reordering)'
    )
    parser.add_argument(
        '--labelmap',
        help='map directory names to paper-ready labels',
        type=str,
        choices=LABELMAP_CHOICES,
        default=None
    )
    parser.add_argument(
        '--csv-name',
        help='name of csv to use for fitness',
        type=str,
        default=DEFAULT_FITNESS_NAME
    )
    args = parser.parse_args()

    plot_multibar_comparison(
        parent_dir=Path(args.parent_dir),
        use_fitness_colors=args.fitness_colors,
        generation=args.generation,
        labelmap=args.labelmap,
        show_best=args.showbest,
        no_legend=args.no_legend,
        legend_order=args.legend_order,
        legend_loc=args.legend_loc,
        normalize_yscores=args.normalize_yscores,
        bar_order=args.bar_order,
        colorblind=args.colorblind,
        csv_name=args.csv_name,
        line_plot_args=parser.dump_line_plot_args(args),
        plot_args=parser.dump_plot_args(args)
    )
