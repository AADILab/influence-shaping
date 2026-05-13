'''Compare how many generations each CCEA method takes to converge across
multiple environmental setups.

parent_dir layout:
  environmental_setup_A/
    method_1/  config.yaml  trial_0/  trial_1/ ...
    method_2/  ...
  environmental_setup_B/
    ...

Each line in the output plot represents one method; each x-tick is one
environmental setup. The y-value is the first generation at which the
averaged (and optionally smoothed) team fitness across stat runs meets or
exceeds the target score.  Points that never converge are plotted at the
maximum generation with a hollow upward-triangle marker.
'''

from pathlib import Path
from influence.plotting import (
    plot_gens_comparison,
    DEFAULT_FITNESS_NAME,
    LEGEND_LOC_CHOICES,
    ENV_ORDER_CHOICES,
)
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='single_gens_comparison.py',
        description='plot generations to convergence across environmental setups',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'parent_dir',
        help='parent directory of environmental setup directories',
        type=str
    )
    parser.add_argument(
        '--fitness-colors',
        help='use defined colors for plotting results of fitness shaping methods',
        action='store_true'
    )
    parser.add_argument(
        '--methods',
        type=str,
        choices=ENV_ORDER_CHOICES,
        default=None,
        help='named preset for ordering and labeling environmental setups on the x-axis (default: alphabetical)'
    )
    parser.add_argument(
        '--score',
        type=float,
        default=None,
        help='target score to measure convergence to (default: maximum attainable score from config.yaml)'
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
        '--log-scale',
        action='store_true',
        help='use log scale for the y axis'
    )
    parser.add_argument(
        '--csv-name',
        help='name of csv to use for fitness',
        type=str,
        default=DEFAULT_FITNESS_NAME
    )
    args = parser.parse_args()

    plot_gens_comparison(
        parent_dir=Path(args.parent_dir),
        use_fitness_colors=args.fitness_colors,
        env_order=args.methods,
        target_score=args.score,
        no_legend=args.no_legend,
        legend_loc=args.legend_loc,
        log_scale=args.log_scale,
        csv_name=args.csv_name,
        line_plot_args=parser.dump_line_plot_args(args),
        plot_args=parser.dump_plot_args(args)
    )
