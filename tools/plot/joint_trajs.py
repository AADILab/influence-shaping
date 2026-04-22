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
        '--num-steps',
        help='number of steps of the joint trajectory to plot',
        type=int
    )
    parser.add_argument(
        '--individual-colors',
        help='plot each agent as a different color',
        action='store_true'
    )
    parser.add_argument(
        '--use-image',
        help='plot image for each agent instead of a marker',
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
    parser.add_argument(
        '--downsample',
        help='only generate one plot for every _ joint trajectories',
        type=int
    )
    parser.add_plot_args()

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    plot_joint_trajectory_tree(
        root_dir=Path(args.root_dir),
        out_dir=out_dir,
        num_steps=args.num_steps,
        individual_colors=args.individual_colors,
        use_image=args.use_image,
        no_poi_shading=args.no_poi_shading,
        no_grid=args.no_grid,
        influence_shading=args.influence_shading,
        uav_observation_radius=args.uav_observation_radius,
        rover_observation_radius=args.rover_observation_radius,
        include_bounds=args.include_bounds,
        downsample=args.downsample,
        batch_plot_args=parser.dump_batch_plot_args(args)
    )
