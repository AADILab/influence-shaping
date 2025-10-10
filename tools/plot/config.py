"""Give this python script a config file and it will plot a map of where pois and agents are"""

from pathlib import Path
from influence.parsing import PlotParser
from influence.plotting import plot_config

if __name__ == "__main__":
    parser = PlotParser(
        prog="config.py",
        description="This plots the map in a given config file (currently limited to fixed placement configs)",
        epilog=""
    )
    parser.add_plot_args()
    parser.add_argument("config_dir")
    parser.add_argument(
        '--individual-colors',
        help='plot each agent as a different color',
        action='store_true'
    )
    parser.add_argument(
        '--no-shading',
        help='turn off shading for poi observation radii',
        action='store_true'
    )
    args = parser.parse_args()

    plot_config(Path(args.config_dir), args.individual_colors, args.no_shading, parser.dump_plot_args(args))
