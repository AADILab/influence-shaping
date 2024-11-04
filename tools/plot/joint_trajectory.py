'''Give this python script a config file and it will plot the specified joint trajectory'''

from pathlib import Path
from influence.plotting import plot_joint_trajectory
from influence.parsing import PlotParser

if __name__ == '__main__':
    parser = PlotParser(
        prog='joint_trajectory.py',
        description='plot the joint trajectory from the specified csv file',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'joint_traj_dir', 
        help='directory of csv file containing joint trajectory',
        type=str
    )
    args = parser.parse_args()

    plot_joint_trajectory(Path(args.joint_traj_dir), parser.dump_plot_args(args))
