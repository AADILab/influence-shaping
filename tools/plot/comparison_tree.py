'''Give this python script the root directory of experiments and it will generate plots
for each lower parameter sweep inside of that directory
'''

from pathlib import Path
from influence.plotting import plot_comparison_tree
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
        type=str
    )
    parser.add_plot_args()

    args = parser.parse_args()

    plot_comparison_tree(Path(args.root_dir), Path(args.out_dir), parser.dump_batch_line_plot_args(args))
