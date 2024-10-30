'''Give this python script a config file and it will plot the specified joint trajectory'''

import argparse
from pathlib import Path
from influence.plotting import plot_joint_trajectory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='joint_trajectory.py',
        description='plot the joint trajectory from the specified csv file',
        epilog=''
    )
    parser.add_argument(
        'joint_traj_dir', 
        help='directory of csv file containing joint trajectory',
        type=str
    )
    parser.add_argument(
        '-s', '--silent',
        help='run silently, without showing the plot',
        action='store_true',
    )
    parser.add_argument(
        '-o', '--output', 
        help='directory to output image of plot to',
        type=str
    )
    args = parser.parse_args()


    if args.output is not None:
        args.output = Path(args.output)

    plot_joint_trajectory(Path(args.joint_traj_dir), args.output, args.silent)
