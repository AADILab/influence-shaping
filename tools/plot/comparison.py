'''Give this python script a comparison directory (parent of multiple parents of trial directories) 
and it will plot the statistics of each parameter combination against each other
'''

from pathlib import Path
from influence.plotting import plot_comparison
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
        '--individual_agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    args = parser.parse_args()

    plot_comparison(Path(args.comparison_dir), parser.dump_line_plot_args(args), parser.dump_plot_args(args))
