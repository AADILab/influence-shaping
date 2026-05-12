'''Give this python script a comparison directory (parent of multiple parents of trial directories)
and it will plot the performance of each parameter combination as a bar chart at a given generation
'''

from pathlib import Path
from influence.plotting import plot_bar_comparison, DEFAULT_FITNESS_NAME, LABELMAP_CHOICES, GROUPING_CHOICES
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='single_bar_comparison.py',
        description='plot bar chart of performance from the specified directory at a given generation',
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
        '--generation',
        help='generation index to snapshot (default: final generation)',
        type=int,
        default=None
    )
    parser.add_argument(
        '--xtick-rotation',
        help='rotation angle in degrees for x-tick labels (default: 45)',
        type=int,
        default=45
    )
    parser.add_argument(
        '--showbest',
        help='draw a dashed line at the highest possible score',
        action='store_true'
    )
    parser.add_argument(
        '--grouping',
        help='cluster bars into named groups with bracket labels below x-axis',
        type=str,
        choices=GROUPING_CHOICES,
        default=None
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

    plot_bar_comparison(
        experiment_dir=Path(args.comparison_dir),
        use_fitness_colors=args.fitness_colors,
        generation=args.generation,
        xtick_rotation=args.xtick_rotation,
        labelmap=args.labelmap,
        grouping=args.grouping,
        show_best=args.showbest,
        csv_name=args.csv_name,
        line_plot_args=parser.dump_line_plot_args(args),
        plot_args=parser.dump_plot_args(args)
    )
