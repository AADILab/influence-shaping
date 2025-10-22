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
    parser.add_argument(
        '--individual-colors',
        help='plot each agent as a different color',
        action='store_true'
    )
    parser.add_argument(
        '--no-poi-shading',
        help='turn off shading for poi observation radii',
        action='store_true'
    )
    parser.add_argument(
        '--no-grid',
        help='turn off grid in background',
        action='store_true'
    )
    # Add shading for radius of influence of uavs
    parser.add_argument(
        '--influence-shading',
        help='turn on shading for radius of influence of uavs',
        action='store_true'
    )
    # Add circle for observation radius of uavs
    parser.add_argument(
        '--uav-observation-radius',
        help='draw circle for observation radius of uavs',
        action='store_true'
    )
    # Add circle for observation radius of rovers
    parser.add_argument(
        '--rover-observation-radius',
        help='draw circle for observation radius of rovers',
        action='store_true'
    )
    # Add bounds for agents
    parser.add_argument(
        '--include-bounds',
        help='draw bounds for each agent',
        action='store_true'
    )
    args = parser.parse_args()

    plot_joint_trajectory(
        Path(args.joint_traj_dir),
        args.individual_colors,
        args.no_poi_shading,
        args.no_grid,
        args.influence_shading,
        args.uav_observation_radius,
        args.rover_observation_radius,
        args.include_bounds,
        parser.dump_plot_args(args)
    )
