'''Give this python script the root directory of experiments and it will generate plots
of each joint trajectory within that root directory
'''

from pathlib import Path
from influence.plotting import plot_joint_trajectory_tree
from influence.parsing import BatchPlotParser

if __name__ == '__main__':
    parser = BatchPlotParser(
        prog='joint_trajectory_tree.py',
        description='plot the joint trajectories in the specified directory',
        epilog=''
    )
    parser.add_argument(
        'root_dir',
        help='root directory with all the joint trajectories',
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
        '--individual_colors',
        help='plot each agent as a different color',
        action='store_true'
    )
    parser.add_argument(
        '--no_shading',
        help='turn off shading for poi observation radii',
        action='store_true'
    )
    parser.add_argument(
        '--downsample',
        help='only generate one plot for every _ joint trajectories',
        action='store_true'
    )
    parser.add_plot_args()

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    plot_joint_trajectory_tree(Path(args.root_dir), out_dir, args.individual_colors, args.no_shading, args.downsample, parser.dump_batch_plot_args(args))
