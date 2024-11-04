'''Give this python script the root directory of experiments and it will generate plots
for each lower parameter sweep inside of that directory
'''

import argparse
from pathlib import Path
from influence.plotting import plot_comparison_tree

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
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

    args = parser.parse_args()

    plot_comparison_tree(Path(args.root_dir), Path(args.out_dir))
