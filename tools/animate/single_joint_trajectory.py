'''Give this python script a config file and it will animate the specified joint trajectory'''

from pathlib import Path
from influence.plotting import plot_joint_trajectory, plot_joint_trajectory_on_ax
from influence.parsing import PlotParser

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update(frame, ax, args, parser):
    ax.clear()
    plot_joint_trajectory_on_ax(
        ax,
        joint_traj_dir=Path(args.joint_traj_dir),
        num_steps=frame,
        individual_colors=args.individual_colors,
        use_image=args.use_image,
        no_shading=args.no_poi_shading,
        no_grid=args.no_grid,
        influence_shading=args.influence_shading,
        uav_observation_radius=args.uav_observation_radius,
        rover_observation_radius=args.rover_observation_radius,
        include_bounds=args.include_bounds,
        plot_args=parser.dump_plot_args(args)
    )

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
    args = parser.parse_args()

    num_frames = sum(1 for _ in open(args.joint_traj_dir)) - 2

    # Derive output path: swap 'results' for 'outvids' in the input path chain.
    # Fall back to outvids/animation.mp4 if no 'results' folder is found.
    csv_path = Path(args.joint_traj_dir).resolve()
    parts = csv_path.parts
    if 'results' in parts:
        idx = parts.index('results')
        out_path = Path(*parts[:idx]) / 'outvids' / Path(*parts[idx+1:])
        out_path = out_path.with_suffix('.mp4')
    else:
        out_path = Path('outvids') / 'animation.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = parser.dump_plot_args(args).init_figure()
    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        fargs=(ax, args, parser),
        interval=100
    )
    ani.save(
        str(out_path),
        writer='ffmpeg',
        fps=30,
        dpi=args.dpi
    )
    plt.show()
