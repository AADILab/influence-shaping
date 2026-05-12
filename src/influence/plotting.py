from typing import List, Optional, Union, Tuple
from collections import OrderedDict
from enum import Enum
import os
import re

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

from influence.config import load_config
from influence.parsing import PlotArgs, LinePlotArgs, BatchPlotArgs, BatchLinePlotArgs

class AgentType(Enum):
    ROVER = 0
    UAV = 1

# Configure matplotlib to use LaTeX fonts
# and nice text sizes
# and a consistent figure size
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,           # Base font size
    "axes.labelsize": 16,      # x and y labels
    "axes.titlesize": 18,      # Title
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 14,     # Legend
    "figure.figsize": (6,4.5)    # Figure size
})

# Get the directory of the current file (e.g., plotting.py)
MODULE_DIR = Path(__file__).parent

# Path to the assets directory (assuming it's at the repo root)
ASSETS_DIR = MODULE_DIR.parent.parent / "assets"

UAV_IMAGE = ASSETS_DIR / "drone.png"
ROVER_IMAGE = ASSETS_DIR / "rover.png"

COMPARISON_COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:purple',
    'tab:olive'
]

COMPARISON_MARKER_MAP = {
    'tab:blue': 's',     # square
    'tab:orange': '^',   # triangle-up
    'tab:green': 'o',    # circle
    'tab:red': 'D',      # diamond
    'tab:purple': 'P',   # plus (filled)
    'tab:brown': 'X',    # x (filled)
    'tab:pink': 'v',     # triangle-down
    'tab:gray': '*',     # star
    'tab:olive': 'd',    # thin diamond
    None: None           # no marker if we don't set the color
}

PASTEL_COLORS = plt.get_cmap('Pastel1').colors
TAB20_COLORS = plt.get_cmap('tab20').colors
COMPARISON_COLORS_DICT = {
    'Global': 'tab:blue',
    'Difference': 'tab:orange',
    'D-Indirect-Traj': 'tab:green',
    'D-Indirect-Traj-Local': TAB20_COLORS[5],
    'D-Indirect-Timestep': 'tab:red',
    'D-Indirect-Timestep-Local': 'tab:purple',
    'D-Indirect-Timestep-System': 'tab:brown',
    'D-Indirect-Timestep-Difference': 'tab:pink',
    'D-I-Sys-uavs-D-rovers': 'tab:gray',
    'G-uavs-D-rovers': 'tab:olive',
    'D-Indirect-Timestep-No-Archive': 'tab:purple',
    'D-Indirect-Traj-No-Archive': 'tab:olive',
    'D-Indirect-Window-N0-n0': TAB20_COLORS[7]
}

JAAMAS_ALL_LABELMAP = {
    'Global': 'Global',
    'Difference': 'Difference',
    'D-Indirect-Traj': 'Static-Competitive',
    'D-Indirect-Traj-Local': 'Static-Loose',
    'D-Indirect-Timestep': 'Dynamic-Competitive',
    'D-Indirect-Window-N0-n0': 'Dynamic-Loose',
}

LABELMAP_CHOICES = ['jaamas-all']

_LABELMAPS = {
    'jaamas-all': JAAMAS_ALL_LABELMAP,
}

def apply_labelmap(label: str, labelmap: Optional[str]) -> str:
    if labelmap is None:
        return label
    mapping = _LABELMAPS.get(labelmap, {})
    if label in mapping:
        return mapping[label]
    # Pattern for adaptive methods: D-Indirect-Window-N{X}-n0, X > 0
    m = re.match(r'D-Indirect-Window-N(\d+)-n0', label)
    if m:
        return f'Adaptive, N={m.group(1)}'
    return label

JAAMAS_SPLIT_GROUPING = OrderedDict([
    ('No Influence',           ['Global', 'Difference']),
    ('Influence Based\n(Implemented)', ['D-Indirect-Traj', 'D-Indirect-Timestep']),
    ('Influence Based\n(Theorized)',             ['D-Indirect-Traj-Local', 'D-Indirect-Window-N0-n0']),
    ('Adaptive + Influence Based',                  [r'D-Indirect-Window-N[1-9]\d*-n\d+']),
])

GROUPING_CHOICES = ['jaamas-split']

_GROUPINGS = {
    'jaamas-split': JAAMAS_SPLIT_GROUPING,
}

def _dir_matches_group_member(dir_name: str, member: str) -> bool:
    if dir_name == member:
        return True
    try:
        return bool(re.fullmatch(member, dir_name))
    except re.error:
        return False

def apply_grouping(dirs: List[Path], grouping: str) -> List[Tuple[str, List[Path]]]:
    """Order dirs by group and return [(group_label, [dirs_in_group]), ...].
    Dirs not matching any group are appended as 'Other'."""
    grouping_config = _GROUPINGS[grouping]
    result = []
    assigned = set()

    for group_name, members in grouping_config.items():
        group_dirs = []
        for dir_ in dirs:
            if dir_.name not in assigned:
                for member in members:
                    if _dir_matches_group_member(dir_.name, member):
                        group_dirs.append(dir_)
                        assigned.add(dir_.name)
                        break
        if group_dirs:
            result.append((group_name, group_dirs))

    ungrouped = [d for d in dirs if d.name not in assigned]
    if ungrouped:
        result.append(('Other', ungrouped))

    return result

LEGEND_LOC_CHOICES = [
    'best',
    'upper right',
    'upper left',
    'lower left',
    'lower right',
    'right',
    'center left',
    'center right',
    'lower center',
    'upper center',
    'center'
]

DEFAULT_FITNESS_NAME = 'fitness.csv'

def compute_markevery(marker_spacing: Union[int, float], num_pts: int) -> int:
    """
    Convert a desired marker spacing into an integer `markevery`
    suitable for matplotlib's index-based marker placement.

    Parameters
    ----------
    marker_spacing : int or float
        - int   -> use directly as markevery
        - float -> fraction of the x-axis domain (0 < f <= 1)
    num_pts : int
        Number of data points in the line

    Returns
    -------
    int
        Integer value for markevery
    """
    if isinstance(marker_spacing, int):
        if marker_spacing <= 0:
            raise ValueError("marker_spacing integer must be > 0")
        return marker_spacing

    if isinstance(marker_spacing, float):
        if not (0.0 < marker_spacing <= 1.0):
            raise ValueError("marker_spacing float must be in (0, 1]")
        if num_pts <= 1:
            raise ValueError("num_pts must be > 1")

        step: int = round(marker_spacing * num_pts)

        # Ensure at least one marker interval
        return max(1, step)

    raise TypeError("marker_spacing must be int or float")

def sort_fitness_path_list(input_list: List[Path]):
    # Filter into fitness shaping names and non-fitness shaping names
    fit_list = []
    nonfit_list = []
    for path in input_list:
        if path.name in COMPARISON_COLORS_DICT:
            fit_list.append(path)
        else:
            nonfit_list.append(path)

    # Now sort the fitness names so they match the COMPARISON_COLORS_DICT list
    # NOTE: This assumes a maximum 1:1 correspondence between fitness shaping method names and directories
    # If for whatever reason this assumption is wrong later, there could be issues.
    sorted_fit_list = []
    for name in COMPARISON_COLORS_DICT:
        for path in fit_list:
            if path.name == name:
                sorted_fit_list.append(path)

    # Sort nonfit list using standard sorting
    nonfit_list.sort()

    return sorted_fit_list+nonfit_list

def get_num_entities_traj(labels: List[str]):
    num_rovers = 0
    num_uavs = 0
    num_hidden_pois = 0
    num_rover_pois = 0

    def get_agent_id(label):
        return int(label.split("_")[-2])

    def get_poi_id(label):
        return int(label.split("_")[-2])

    for label in labels:
        if "rover" in label and "_x" in label and "poi" not in label:
            id_ = get_agent_id(label)
            if id_ + 1 > num_rovers:
                num_rovers += 1
        elif "uav" in label and "_x" in label:
            id_ = get_agent_id(label)
            if id_ + 1 > num_uavs:
                num_uavs += 1
        elif "rover_poi" in label and "_x" in label:
            id_ = get_poi_id(label)
            if id_ + 1 > num_rover_pois:
                num_rover_pois += 1
        elif "hidden_poi" in label and "_x" in label:
            id_ = get_poi_id(label)
            if id_ + 1 > num_hidden_pois:
                num_hidden_pois += 1

    return num_rovers, num_uavs, num_rover_pois, num_hidden_pois

def get_num_entities_fit(labels: List[str]):
    num_rovers = 0
    num_uavs = 0
    num_hidden_pois = 0
    num_rover_pois = 0

    def get_agent_id(label):
        return int(label.split("_")[-1])

    def get_poi_id(label):
        return int(label.split("_")[-1])

    for label in labels:
        if "rover" in label and "poi" not in label:
            id_ = get_agent_id(label)
            if id_ + 1 > num_rovers:
                num_rovers += 1
        elif "uav" in label:
            id_ = get_agent_id(label)
            if id_ + 1 > num_uavs:
                num_uavs += 1
        elif "rover_poi" in label:
            id_ = get_poi_id(label)
            if id_ + 1 > num_rover_pois:
                num_rover_pois += 1
        elif "hidden_poi" in label:
            id_ = get_poi_id(label)
            if id_ + 1 > num_hidden_pois:
                num_hidden_pois += 1

    return num_rovers, num_uavs, num_rover_pois, num_hidden_pois

def plot_poi(ax, poi_config, x, y, color, radius_shading):
    center_circle = plt.Circle(
        xy = (x, y),
        radius = min(1.0, poi_config['capture_radius']),
        facecolor=color,
        edgecolor='none',
        fill=True,
        alpha=1.0
        )
    ax.add_patch(center_circle)
    if radius_shading:
        outer_circle = plt.Circle(
            xy = (x, y),
            radius = poi_config['capture_radius'],
            facecolor=color,
            edgecolor='none',
            fill=True,
            alpha=0.2
            )
        ax.add_patch(outer_circle)

def get_rover_colors(individual_colors: bool):
    if individual_colors:
        rover_colors = plt.cm.Set1.colors[:1]+plt.cm.Set1.colors[3:]
    else:
        rover_colors = [(162/255, 197/255, 202/255)]
    return rover_colors

def get_uav_colors(individual_colors: bool):
    if individual_colors:
        uav_colors = plt.cm.Dark2.colors[1:]
    else:
        uav_colors = [(114/255, 38/255, 115/255)]
    return uav_colors

def add_trajectory(
        ax: Axes,
        xs: np.ndarray,
        ys: np.ndarray,
        color: tuple,
        use_image: bool,
        influence_shading: bool,
        agent_type: AgentType,
        observation_radius: Optional[float],
        bounds: Optional[List[Optional[dict]]],
        num_steps: Optional[int]
    ):
    # Cut down xs and ys according to num steps
    if num_steps is not None:
        xs = xs[0:num_steps+1]
        ys = ys[0:num_steps+1]

    # Plot the trajectory "trace" dots
    ax.plot(xs, ys, linestyle='None', color=color, marker='o', markersize=0.5, alpha=0.75)

    # Place the marker / image at the final position
    x_final = xs[-1]
    y_final = ys[-1]
    if use_image:
        # Set agent specific parameters
        if agent_type == AgentType.ROVER:
            image_path = ROVER_IMAGE
            orig_color = np.array([162, 197, 202, 255]) # RGBA
            # Get last two positions for heading calculation
            if len(xs) >= 2:
                x_prev, y_prev = xs[-2], ys[-2]
            else:
                x_prev, y_prev = x_final, y_final
            # Compute angle in degrees (0 = right, 90 = up, etc.)
            dx = x_final - x_prev
            dy = y_final - y_prev
            if dx == 0 and dy == 0:
                rotation = 0
            else:
                angle_rad = np.arctan2(dy, dx)
                rotation = np.degrees(angle_rad)
        else:
            image_path = UAV_IMAGE
            orig_color = np.array([114, 38, 115, 255]) # RGBA
            rotation = 0

        # Grab the appropriate image
        img = Image.open(str(image_path))
        img_arr = np.array(img)

        # Swap the color. Convert back to PIL Image
        target_rgb = np.array([int(255*c) if c <= 1 else int(c) for c in color])
        target_color = np.array([*target_rgb, 255]) # RGBA
        mask = np.all(img_arr[:, :, :3] == orig_color[:3], axis=-1)
        img_arr[mask] = target_color
        img = Image.fromarray(img_arr)

        # Rotate PIL image according to heading. Convert to np array.
        if rotation != 0:
            img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)
        img = np.array(img)

        # Add the image to the ax object
        zoom = 0.01
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x_final, y_final), frameon=False)
        ax.add_artist(ab)
    else:
        if agent_type == AgentType.ROVER:
            marker_type = 's'
        else:
            marker_type = 'x'
        ax.plot(x_final, y_final, marker_type, ms=6, color=color)

    if observation_radius is not None:
        observation_circle = plt.Circle(
            xy=(x_final,y_final),
            radius=observation_radius,
            color=color,
            fill=False,
            linewidth=1
        )
        ax.add_patch(observation_circle)

    if bounds is not None:
        low_x = bounds['low_x']
        high_x = bounds['high_x']
        low_y = bounds['low_y']
        high_y = bounds['high_y']
        bounds_rect = plt.Rectangle(
            xy=(low_x,low_y),
            width=high_x-low_x,
            height=high_y-low_y,
            color=color,
            fill=False,
            linewidth=1,
            linestyle=':'
        )
        ax.add_patch(bounds_rect)

    if influence_shading and agent_type == AgentType.ROVER:
        influence_circle = plt.Circle(
            xy = (x_final, y_final),
            radius = 5.0,
            facecolor= color,
            fill=True,
            alpha=0.25,
            linewidth=0,
            edgecolor='none'
        )
        ax.add_patch(influence_circle)

def add_trajectories(
        ax: Axes,
        df: pd.DataFrame,
        num_rovers: int,
        num_uavs: int,
        individual_colors: bool,
        use_image: bool,
        influence_shading: bool,
        rover_observation_radii: Optional[List[float]],
        uav_observation_radii: Optional[List[float]],
        rover_bounds_list: Optional[List[Optional[dict]]],
        uav_bounds_list: Optional[List[Optional[dict]]],
        num_steps: Optional[int]
    ):
    # Setting defaults
    bounds = None
    observation_radius = None
    # Add the rover trajectories
    rover_colors = get_rover_colors(individual_colors)
    for i in range(num_rovers):
        color = rover_colors[i%len(rover_colors)]
        xs = np.array(df['rover_'+str(i)+'_x'])
        ys = np.array(df['rover_'+str(i)+'_y'])
        if rover_observation_radii is not None:
            observation_radius = rover_observation_radii[i]
        if rover_bounds_list is not None:
            bounds = rover_bounds_list[i]
        add_trajectory(
            ax=ax,
            xs=xs,
            ys=ys,
            color=color,
            use_image=use_image,
            influence_shading=influence_shading,
            agent_type=AgentType.ROVER,
            observation_radius=observation_radius,
            bounds=bounds,
            num_steps=num_steps
        )
    # Add the uav trajectories
    uav_colors = get_uav_colors(individual_colors)
    for i in range(num_uavs):
        color = uav_colors[i%len(uav_colors)]
        xs = np.array(df['uav_'+str(i)+'_x'])
        ys = np.array(df['uav_'+str(i)+'_y'])
        if uav_observation_radii is not None:
            observation_radius = uav_observation_radii[i]
        if uav_bounds_list is not None:
            bounds = uav_bounds_list[i]
        add_trajectory(
            ax=ax,
            xs=xs,
            ys=ys,
            color=color,
            use_image=use_image,
            influence_shading=influence_shading,
            agent_type=AgentType.UAV,
            observation_radius=observation_radius,
            bounds=bounds,
            num_steps=num_steps
        )

def plot_joint_trajectory_on_ax(
        ax,
        joint_traj_dir: Path,
        num_steps: Optional[int],
        individual_colors: bool,
        use_image: bool,
        no_shading: bool,
        no_grid: bool,
        influence_shading: bool,
        uav_observation_radius: bool,
        rover_observation_radius: bool,
        include_bounds: bool,
        plot_args: PlotArgs
    ):
    if not no_grid:
        ax.grid(zorder=0)
        ax.set_axisbelow(True)

    # Get the joint trajectory
    df = pd.read_csv(joint_traj_dir)

    # Get config for map bounds by crawling up the directory tree
    config_dir = None
    current_dir = joint_traj_dir.parent
    while current_dir != current_dir.parent:  # Stop at root directory
        potential_config = current_dir / 'config.yaml'
        if potential_config.exists():
            config_dir = potential_config
            break
        current_dir = current_dir.parent

    if config_dir is None:
        raise FileNotFoundError(f"No config.yaml found in any parent directory of {joint_traj_dir}")

    config = load_config(config_dir)
    rover_configs = config['env']['agents']['rovers']
    uav_configs = config['env']['agents']['uavs']

    # Get the number of each entity
    num_rovers, num_uavs, _, _ \
        = get_num_entities_traj(labels=df.columns.to_list())

    # Collect observation radii if specified
    rover_observation_radii = None
    if rover_observation_radius:
        rover_observation_radii = [
            rover_config['observation_radius'] for rover_config in rover_configs
        ]
    uav_observation_radii = None
    if uav_observation_radius:
        uav_observation_radii = [
            uav_config['observation_radius'] for uav_config in uav_configs
        ]

    # Collect bounds if specified
    all_rover_bounds = None
    all_uav_bounds = None
    if include_bounds:
        all_rover_bounds = []
        for rover_config in rover_configs:
            if 'bounds' in rover_config:
                all_rover_bounds.append(rover_config['bounds'])
            else:
                all_rover_bounds.append(None)
        all_uav_bounds = []
        for uav_config in uav_configs:
            if 'bounds' in uav_config:
                all_uav_bounds.append(uav_config['bounds'])
            else:
                all_uav_bounds.append(None)

    add_trajectories(
        ax=ax,
        df=df,
        num_rovers=num_rovers,
        num_uavs=num_uavs,
        individual_colors=individual_colors,
        use_image=use_image,
        influence_shading=influence_shading,
        rover_observation_radii=rover_observation_radii,
        uav_observation_radii=uav_observation_radii,
        rover_bounds_list=all_rover_bounds,
        uav_bounds_list=all_uav_bounds,
        num_steps=num_steps
    )

    for i, poi_config in enumerate(config['env']['pois']['rover_pois']):
        plot_poi(ax, poi_config, x=df['rover_poi_'+str(i)+'_x'][0], y=df['rover_poi_'+str(i)+'_y'][0], color='tab:green', radius_shading=not no_shading)
    for i, poi_config in enumerate(config['env']['pois']['hidden_pois']):
        plot_poi(ax, poi_config, x=df['hidden_poi_'+str(i)+'_x'][0], y=df['hidden_poi_'+str(i)+'_y'][0], color='tab:green', radius_shading=not no_shading)

    x_bound, y_bound = config['env']['map_size']

    ax.set_xlim([0, x_bound])
    ax.set_ylim([0, y_bound])
    ax.set_aspect('equal')

    plot_args.apply(ax)


def generate_joint_trajectory_plot(
        joint_traj_dir: Path,
        num_steps: Optional[int],
        individual_colors: bool,
        use_image: bool,
        no_shading: bool,
        no_grid: bool,
        influence_shading: bool,
        uav_observation_radius: bool,
        rover_observation_radius: bool,
        include_bounds: bool,
        plot_args: PlotArgs
    ):
    """Generate plot of the joint trajectory specified in joint_traj_dir"""

    fig, ax = plot_args.init_figure()
    plot_joint_trajectory_on_ax(
        ax,
        joint_traj_dir=joint_traj_dir,
        num_steps=num_steps,
        individual_colors=individual_colors,
        use_image=use_image,
        no_shading=no_shading,
        no_grid=no_grid,
        influence_shading=influence_shading,
        uav_observation_radius=uav_observation_radius,
        rover_observation_radius=rover_observation_radius,
        include_bounds=include_bounds,
        plot_args=plot_args
    )

    return fig

def plot_joint_trajectory(
        joint_traj_dir: Path,
        num_steps: Optional[int],
        individual_colors: bool,
        use_image: bool,
        no_poi_shading: bool,
        no_grid: bool,
        influence_shading: bool,
        uav_observation_radius: bool,
        rover_observation_radius: bool,
        include_bounds: bool,
        plot_args: PlotArgs
    ):
    fig = generate_joint_trajectory_plot(
        joint_traj_dir,
        num_steps,
        individual_colors,
        use_image,
        no_poi_shading,
        no_grid,
        influence_shading,
        uav_observation_radius,
        rover_observation_radius,
        include_bounds,
        plot_args
    )
    plot_args.finish_figure(fig)

def add_learning_curve(
        ax: Axes,
        df: pd.DataFrame,
        line_plot_args: LinePlotArgs,
        label: str = 'team'
    ):
    """Add the team's learning curve from the specified fitness directory to the Axes object"""

    # Get the points for plotting team fitness
    # print(df['generation'])
    gens, fits = line_plot_args.get_pts(xs=df['generation'], ys=df['collapsed_team_fitness'])
    # print(gens)
    ax.plot(gens, fits, label=label)

    return gens

def generate_learning_curve_plot(
        fitness_dir,
        individual_agents,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    """Generate plot of the learning curve specified in fitness_dir"""

    fig, ax = plot_args.init_figure()

    # Get the fitnesses
    df = pd.read_csv(fitness_dir)
    # print(df['generation'])

    # Get points for plotting team fitness
    gens = add_learning_curve(ax, df, line_plot_args)

    if individual_agents:
        num_rovers, num_uavs, _, _ = get_num_entities_fit(labels=df.columns.to_list())
        for i in range(num_rovers):
            rover_label = 'collapsed_rover_'+str(i)
            fits = line_plot_args.get_ys(ys=df[rover_label])
            ax.plot(gens, fits, label=rover_label)
        for i in range(num_uavs):
            uav_label = 'collapsed_uav_'+str(i)
            fits = line_plot_args.get_ys(ys=df[uav_label])
            ax.plot(gens, fits, label=uav_label)
        ax.legend()

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens.iloc[-1]])
    config = load_config(fitness_dir.parent.parent/'config.yaml')
    high_y = sum([poi_config['value'] for poi_config in config['env']['pois']['hidden_pois']+config['env']['pois']['rover_pois']])
    ax.set_ylim([0, high_y])

    plot_args.apply(ax)

    return fig

def plot_learning_curve(
        fitness_dir: Path,
        individual_agents: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    fig = generate_learning_curve_plot(fitness_dir, individual_agents, line_plot_args, plot_args)
    plot_args.finish_figure(fig)

def add_stat_learning_curve(
        ax: Axes,
        individual_trials: bool,
        csv_name: str,
        trials_dir: Path,
        label: str,
        line_plot_args: LinePlotArgs,
        color: Optional[Union[str,Tuple[float]]] = None,
        marker: Optional[str] = None
    ):
    # Set default color and marker
    if color is None:
        color = COMPARISON_COLORS[0]
    if marker is None:
        marker = COMPARISON_MARKER_MAP[color]

    # Get the directories of trials
    dirs = [trials_dir/dir for dir in os.listdir(trials_dir) if 'trial_' in dir]

    # Sort directories by trial number
    dirs.sort(key=lambda x: int(str(x).split('_')[-1]))

    # Get the fitnesses in each trial
    dfs = [pd.read_csv(dir/csv_name) for dir in dirs]

    # Plot individual trials if specified
    if individual_trials:
        # Plot each trial's fitness throughout training
        for df, dir in zip(dfs, dirs):
            gens = add_learning_curve(ax, df, line_plot_args, label=dir.name)

        # Put gens in the expected format
        gens = np.array(gens)

        ax.legend()

        return gens

    # Otherwise plot mean and standard error
    else:
        # Figure out which trial ran the shortest
        # (We can only accurately compute statistics for generations that we have all trials' output for)
        ind = min([len(df['collapsed_team_fitness']) for df in dfs])

        # Compute the statistics
        avg = np.average([df['collapsed_team_fitness'][:ind] for df in dfs], axis=0)
        err = np.std([df['collapsed_team_fitness'][:ind] for df in dfs], axis=0) / np.sqrt(len(dfs))
        upp_err = avg+err
        low_err = avg-err
        gens = list(range(len(avg)))

        # Clean up data
        gens, avg = line_plot_args.get_pts(gens, avg)
        low_err = line_plot_args.get_ys(low_err)
        upp_err = line_plot_args.get_ys(upp_err)
        # print('stats for ', trials_dir, ' | avg[1,000]: ', avg[1000], ' | avg[-1]: ', avg[-1])

        # Compute marker spacing
        markevery=compute_markevery(marker_spacing=0.1, num_pts=len(gens))

        # Plot statistics
        ax.plot(
            gens,
            avg,
            label=label,
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=10
        )
        ax.fill_between(
            gens,
            low_err,
            upp_err,
            alpha=0.2,
            facecolor=color
        )

        # Set ax ylim based on poi values in config
        config = load_config(trials_dir/'config.yaml')
        high_y = sum(
            poi_config['value'] for poi_config in config['env']['pois']['hidden_pois']+config['env']['pois']['rover_pois']
        )
        ax.set_ylim([0, high_y])

        return gens

def generate_stat_learning_curve_plot(
        trials_dir: Path,
        individual_trials: bool,
        csv_name: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    """Generate plot of statistics of learning given the parent directoy of trials"""

    fig, ax = plot_args.init_figure()

    gens = add_stat_learning_curve(ax, individual_trials, csv_name, trials_dir, label=trials_dir.name, line_plot_args=line_plot_args)

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens[-1]])
    # Set the y limit based on the values of pois in the config
    config = load_config(trials_dir/'config.yaml')
    high_y = sum([poi_config['value'] for poi_config in config['env']['pois']['hidden_pois']+config['env']['pois']['rover_pois']])
    ax.set_ylim([0, high_y])

    plot_args.apply(ax)

    return fig

def plot_stat_learning_curve(
        trials_dir,
        individual_trials,
        csv_name,
        line_plot_args,
        plot_args
    ):
    fig = generate_stat_learning_curve_plot(
        trials_dir,
        individual_trials,
        csv_name,
        line_plot_args,
        plot_args
    )
    plot_args.finish_figure(fig)

def generate_stat_learning_curve_tree_plots(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        individual_trials: bool = False,
        csv_name: str = DEFAULT_FITNESS_NAME,
        batch_plot_args: BatchPlotArgs = None,
        batch_line_plot_args: BatchLinePlotArgs = None
    ):
    """Generate all the stat learning curve plots in this experiment tree"""

    if out_dir is None:
        # Infer out_dir by replacing 'results' with 'outfigs' in root_dir
        root_str = str(root_dir)
        if 'results' not in root_str:
            raise ValueError("No 'results' folder found in root_dir. out_dir must be specified so plots can be saved somewhere")
        out_dir = Path(root_str.replace('results', 'outfigs'))

    experiment_dirs = set()
    for root, _, files in os.walk(root_dir):
        if 'config.yaml' in files:
            experiment_dirs.add(Path(root))

    for dir_ in experiment_dirs:
        dir_list = str(dir_).split('/')
        dir_name = '/'.join(dir_list[dir_list.index(root_dir.name)+1:])

        file_append = ''
        if csv_name != DEFAULT_FITNESS_NAME:
            file_append+='.'+csv_name.split('.')[0]
        if individual_trials:
            file_append+='.ind'
        if batch_line_plot_args.window_size is not None:
            file_append+='.w'+str(batch_line_plot_args.window_size)

        plot_stat_learning_curve(
            trials_dir=dir_,
            individual_trials=individual_trials,
            csv_name=csv_name,
            line_plot_args=batch_line_plot_args.build_line_plot_args(),
            plot_args=batch_plot_args.build_plot_args(
                title=dir_name, output=out_dir/dir_name/('stat_learning_curve'+file_append+'.png')
            )
        )

def plot_stat_learning_curve_tree(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        individual_trials: bool = False,
        csv_name: str = DEFAULT_FITNESS_NAME,
        batch_plot_args: BatchPlotArgs = None,
        batch_line_plot_args: BatchLinePlotArgs = None
    ):
    generate_stat_learning_curve_tree_plots(
        root_dir,
        out_dir,
        individual_trials,
        csv_name,
        batch_plot_args,
        batch_line_plot_args
    )

def sort_legend(ax, legend_order):
    if legend_order is None:
        handles, labels = ax.get_legend_handles_labels()
        return handles, labels

    elif legend_order == 'acm-telo':
        # Define label name mapping
        label_name_map = {
            'D-Indirect-Timestep': r'Dynamic Influence, $D^{DYN}$',
            'D-Indirect-Traj': r'Static Influence, $D^{IND}$',
            'Global': r'Team Fitness, $G$',
            'Difference': r'Difference Fitness, $D$',
            'D-Indirect-Timestep-No-Archive': r'[No Archive] Dynamic Influence',
            'D-Indirect-Traj-No-Archive': r'[No Archive] Static Influence'
        }

        # Define desired order
        desired_order = [
            'D-Indirect-Timestep',
            'D-Indirect-Traj',
            'D-Indirect-Timestep-No-Archive',
            'D-Indirect-Traj-No-Archive',
            'Global',
            'Difference'
        ]

    elif legend_order == 'jaamas':
        # Define label name mapping
        label_name_map = {
            'D-Indirect-Timestep': r'Dynamic',
            'D-Indirect-Traj': r'Static',
            'Global': r'Global',
            'Difference': r'Direct',
            'D-Indirect-Window-N1-n0': r'Adaptive, N=1'
        }

        # Define desired order
        desired_order = [
            'D-Indirect-Window-N1-n0',
            'D-Indirect-Traj',
            'Global',
            'D-Indirect-Timestep',
            'Difference'
        ]

    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Reorder handles and labels
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        if desired_label in labels:
            idx = labels.index(desired_label)
            ordered_handles.append(handles[idx])
            # Use mapped name if available, otherwise use original
            ordered_labels.append(label_name_map.get(desired_label, desired_label))

    # Add any remaining labels not in desired_order
    for handle, label in zip(handles, labels):
        if label not in desired_order:
            ordered_handles.append(handle)
            # Use mapped name if available, otherwise use original
            ordered_labels.append(label_name_map.get(label, label))

    return ordered_handles, ordered_labels

def generate_comparison_plot(
        experiment_dir: Path,
        use_fitness_colors: bool,
        legend_order: Optional[str],
        legend_loc: Optional[str],
        no_legend: bool,
        csv_name: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    """Generate plot of experiment using experiment directory
    experiment_dir is parent of parent of trial directories
    """
    # print(experiment_dir)
    fig, ax = plot_args.init_figure()

    # Set background color and grid style
    ax.set_facecolor('#e6e6e6')
    ax.grid(True, color='white', linewidth=1.0)
    ax.set_axisbelow(True)

    # Get the parent dirs of trials
    dirs = [experiment_dir/dir for dir in os.listdir(experiment_dir)]

    # If we are using the fitness color set, then sort the methods so order is always consistent
    sorted_dirs = sort_fitness_path_list(dirs)

    xlim = 0
    for i, trials_dir in enumerate(sorted_dirs):
        color=None
        if use_fitness_colors and trials_dir.name in COMPARISON_COLORS_DICT:
            # Set color based on fitness shaping method (optional)
            # Use extra colors for names that have not been reserved
            color=COMPARISON_COLORS_DICT[trials_dir.name]
        else:
            color = COMPARISON_COLORS[(i+len(COMPARISON_COLORS_DICT))%len(COMPARISON_COLORS)]

        # Add the marker associated with this color (if there is one)
        # print(COMPARISON_MARKER_MAP, color)
        marker=COMPARISON_MARKER_MAP[color]
        gens = add_stat_learning_curve(
            ax,
            False,
            csv_name,
            trials_dir,
            label=trials_dir.name,
            line_plot_args=line_plot_args,
            color=color,
            marker=marker
        )

        if gens[-1] > xlim:
            xlim = gens[-1]

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    # Sort the legend according to the specified order (optional)
    handles, labels = sort_legend(ax, legend_order)

    # Place the legend
    if not no_legend:
        ax.legend(handles, labels, loc=legend_loc)

    ax.set_xlim([0, gens[-1]])

    plot_args.apply(ax)

    return fig

def plot_comparison(
        experiment_dir: Path,
        use_fitness_colors: bool,
        legend_order: Optional[str],
        legend_loc: Optional[str],
        no_legend: bool,
        csv_name: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    fig = generate_comparison_plot(
        experiment_dir=experiment_dir,
        use_fitness_colors=use_fitness_colors,
        legend_order=legend_order,
        legend_loc=legend_loc,
        no_legend=no_legend,
        csv_name=csv_name,
        line_plot_args=line_plot_args,
        plot_args=plot_args
    )
    plot_args.finish_figure(fig)

def get_bar_snapshot(
        trials_dir: Path,
        csv_name: str,
        generation: Optional[int],
        line_plot_args: LinePlotArgs
    ) -> Tuple[Optional[float], Optional[float]]:
    dirs = [trials_dir/dir for dir in os.listdir(trials_dir) if 'trial_' in dir]
    if not dirs:
        return None, None
    dirs.sort(key=lambda x: int(str(x).split('_')[-1]))
    dfs = [pd.read_csv(dir/csv_name) for dir in dirs if (dir/csv_name).exists()]
    if not dfs:
        return None, None

    ind = min(len(df['collapsed_team_fitness']) for df in dfs)

    fits_at_snapshot = []
    for df in dfs:
        fits = np.array(df['collapsed_team_fitness'][:ind])
        fits = line_plot_args.get_ys(fits)
        snap_idx = generation if generation is not None else -1
        fits_at_snapshot.append(float(fits[snap_idx]))

    avg = float(np.mean(fits_at_snapshot))
    err = float(np.std(fits_at_snapshot) / np.sqrt(len(fits_at_snapshot)))
    return avg, err

def generate_bar_comparison_plot(
        experiment_dir: Path,
        use_fitness_colors: bool,
        generation: Optional[int],
        xtick_rotation: int,  # adjust bar label rotation here
        labelmap: Optional[str],
        grouping: Optional[str],
        show_best: bool,
        csv_name: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    from matplotlib.transforms import blended_transform_factory

    fig, ax = plot_args.init_figure()

    ax.set_facecolor('#e6e6e6')
    ax.grid(True, color='white', linewidth=1.0, axis='y')
    ax.set_axisbelow(True)

    dirs = [experiment_dir/dir for dir in os.listdir(experiment_dir)]
    sorted_dirs = sort_fitness_path_list(dirs)

    config = load_config(sorted_dirs[0] / 'config.yaml')
    high_y = sum(
        poi_config['value'] for poi_config in
        config['env']['pois']['hidden_pois'] + config['env']['pois']['rover_pois']
    )

    # Determine draw order and x positions
    # Within-group bar spacing (center-to-center) and extra gap between groups
    BAR_SPACING = 1.0
    GROUP_GAP   = 0.8

    if grouping is not None:
        grouped = apply_grouping(sorted_dirs, grouping)
        ordered_dirs = []
        x_positions = []
        group_spans = []  # (group_label, x_first_bar, x_last_bar)
        x_cursor = 0.0
        for group_name, group_dirs in grouped:
            x_first = x_cursor
            for dir_ in group_dirs:
                x_positions.append(x_cursor)
                ordered_dirs.append(dir_)
                x_cursor += BAR_SPACING
            x_last = x_cursor - BAR_SPACING
            group_spans.append((group_name, x_first, x_last))
            x_cursor += GROUP_GAP
        fig.subplots_adjust(top=0.78)
    else:
        ordered_dirs = sorted_dirs
        x_positions = list(np.arange(len(sorted_dirs), dtype=float))
        group_spans = []

    # Draw bars
    for i, (trials_dir, x_pos) in enumerate(zip(ordered_dirs, x_positions)):
        if use_fitness_colors and trials_dir.name in COMPARISON_COLORS_DICT:
            color = COMPARISON_COLORS_DICT[trials_dir.name]
        else:
            color = COMPARISON_COLORS[(i + len(COMPARISON_COLORS_DICT)) % len(COMPARISON_COLORS)]

        avg, err = get_bar_snapshot(trials_dir, csv_name, generation, line_plot_args)

        if avg is None:
            ax.bar(x_pos, high_y * 0.8, color='none', edgecolor='gray', linewidth=1.5, linestyle='--')
            ax.text(
                x_pos, high_y * 0.02, 'Results pending',
                ha='center', va='bottom',
                rotation=90, fontsize=10, color='gray', style='italic'
            )
        else:
            ax.bar(x_pos, avg, yerr=err, color=color, capsize=5, error_kw={'linewidth': 1.5})

    # Bar labels (labelmap applied here, after grouping order is fixed)
    labels = [apply_labelmap(d.name, labelmap) for d in ordered_dirs]
    ax.set_xticks(x_positions)
    ha = 'right' if xtick_rotation != 0 else 'center'
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha=ha)

    # Draw group brackets above bars
    if group_spans:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        half_bar = 0.4  # matches default matplotlib bar width of 0.8
        for group_label, x_first, x_last in group_spans:
            x_lo = x_first - half_bar
            x_hi = x_last  + half_bar
            x_mid = (x_first + x_last) / 2
            # Horizontal bracket line
            ax.plot([x_lo, x_hi], [1.06, 1.06],
                    transform=trans, color='black', lw=1.0, clip_on=False)
            # End ticks pointing down from the line
            ax.plot([x_lo, x_lo], [1.02, 1.06],
                    transform=trans, color='black', lw=1.0, clip_on=False)
            ax.plot([x_hi, x_hi], [1.02, 1.06],
                    transform=trans, color='black', lw=1.0, clip_on=False)
            # Group label above the line
            ax.text(x_mid, 1.09, group_label,
                    transform=trans, ha='center', va='bottom',
                    clip_on=False, fontsize=11, style='italic')

    ax.set_ylabel('Performance')
    ax.set_ylim([0, high_y])

    if show_best:
        ax.axhline(y=high_y, color='black', linestyle='--', linewidth=1.5)
        text_trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.05, high_y * 1.06, 'Highest Possible Score',
                transform=text_trans, ha='left', va='top', color='black', fontsize=12)

    plot_args.apply(ax)
    return fig

def plot_bar_comparison(
        experiment_dir: Path,
        use_fitness_colors: bool,
        generation: Optional[int],
        xtick_rotation: int,
        labelmap: Optional[str],
        grouping: Optional[str],
        show_best: bool,
        csv_name: str,
        line_plot_args: LinePlotArgs,
        plot_args: PlotArgs
    ):
    fig = generate_bar_comparison_plot(
        experiment_dir=experiment_dir,
        use_fitness_colors=use_fitness_colors,
        generation=generation,
        xtick_rotation=xtick_rotation,
        labelmap=labelmap,
        grouping=grouping,
        show_best=show_best,
        csv_name=csv_name,
        line_plot_args=line_plot_args,
        plot_args=plot_args
    )
    plot_args.finish_figure(fig)

def get_example_trial_dirs(parent_dir: Path):
    dirs = [parent_dir/dir for dir in os.list(parent_dir) if 'trial_' in dir]
    dfs = [pd.read_csv(dir/'fitness.csv') for dir in dirs]
    fits = [df['collapsed_team_fitness'] for df in dfs]
    final_fits = [fit[-1] for fit in fits]
    class FitPair():
        def __init__(self, fit, ind):
            self.fit = fit
            self.ind = ind
    fit_inds = [FitPair(fit, ind) for ind, fit in enumerate(final_fits)]
    # Sort according to the fitness value
    fit_inds.sort(lambda x: x.fit)

    # Now get the index of low, medium, and high performers
    high_ind = fit_inds[-1].ind
    med_ind = fit_inds[len(fit_inds)/2].ind
    low_ind = fit_inds[0].ind

    # Turn that into trials
    low_trial_dir = parent_dir / ('trial_'+str(low_ind))
    med_trial_dir = parent_dir / ('trial_'+str(med_ind))
    high_trial_dir = parent_dir / ('trial_'+str(high_ind))

    return low_trial_dir, med_trial_dir, high_trial_dir

def generate_experiment_tree_plots(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        use_fitness_colors: bool = False,
        legend_order: Optional[str] = None,
        legend_loc: Optional[str] = None,
        no_legend: bool = False,
        csv_name: str = DEFAULT_FITNESS_NAME,
        batch_plot_args: BatchPlotArgs = None,
        batch_line_plot_args: BatchLinePlotArgs = None
    ):
    """Generate all the plots in this experiment tree"""

    if out_dir is None:
        # Infer out_dir by replacing 'results' with 'outfigs' in root_dir
        root_str = str(root_dir)
        if 'results' not in root_str:
            raise ValueError("No 'results' folder found in root_dir. out_dir must be specified so plots can be saved somewhere")
        out_dir = Path(root_str.replace('results', 'outfigs'))

    experiment_dirs = set()
    trial_parent_dirs = set()
    for root, _, files in os.walk(root_dir):
        if 'config.yaml' in files:
            experiment_dirs.add(Path(root).parent)
            trial_parent_dirs.add(Path(root))

    for dir_ in experiment_dirs:
        dir_list = str(dir_).split("/")
        dir_name = "/".join(dir_list[dir_list.index(root_dir.name)+1:])

        file_append = ''
        if csv_name != DEFAULT_FITNESS_NAME:
            file_append+='.'+csv_name.split('.')[0]
        if batch_line_plot_args.window_size is not None:
            file_append+='.w'+str(batch_line_plot_args.window_size)

        plot_comparison(
            experiment_dir=dir_,
            use_fitness_colors=use_fitness_colors,
            legend_order=legend_order,
            legend_loc=legend_loc,
            no_legend=no_legend,
            csv_name=csv_name,
            line_plot_args=batch_line_plot_args.build_line_plot_args(),
            plot_args=batch_plot_args.build_plot_args(
                title=dir_name, output=out_dir/dir_name/('comparison'+file_append+'.png')
            )
        )

def sort_jt_dirs(root_dir: Path, jt_dirs: List[str]):
    # Starting place for sorting dirs for joint trajectories

    root_len = len(str(root_dir).split('/'))
    sort_jt_dirs_helper(jt_dirs, level=root_len)

def sort_jt_dirs_helper(jt_dirs: List[str], level: int):
    # Sort the specified level, then pass it on
    # Everything happens in place - this is recursive, but it's a linear operation

    # Base cases: if we are the trials level, then use a special lambda function for that
    # If we are at the gens, level, use the same lambda function for that
    if 'trial_' in jt_dirs[0].split('/')[level]:
        jt_dirs.sort(key = lambda x: int(x.split('/')[level].split('_')[-1]))
        sort_jt_dirs_helper(jt_dirs, level=level+1)

    elif 'gen_' in jt_dirs[0].split('/')[level]:
        jt_dirs.sort(key = lambda x: int(x.split('/')[level].split('_')[-1]))

    # General case. Sort and keep going
    else:
        jt_dirs.sort(key = lambda x: x.split('/')[level])
        sort_jt_dirs_helper(jt_dirs, level=level+1)

def plot_comparison_tree(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        use_fitness_colors: bool = False,
        legend_order: Optional[str] = None,
        legend_loc: Optional[str] = None,
        no_legend: bool = False,
        csv_name: str = DEFAULT_FITNESS_NAME,
        batch_plot_args: BatchPlotArgs = None,
        batch_line_plot_args: BatchLinePlotArgs = None
    ):
    generate_experiment_tree_plots(
        root_dir=root_dir,
        out_dir=out_dir,
        use_fitness_colors=use_fitness_colors,
        legend_order=legend_order,
        legend_loc=legend_loc,
        no_legend=no_legend,
        csv_name=csv_name,
        batch_plot_args=batch_plot_args,
        batch_line_plot_args=batch_line_plot_args
    )

def generate_joint_trajectory_tree_plots(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        num_steps: Optional[int] = None,
        individual_colors: bool = False,
        use_image: bool = False,
        no_shading: bool = False,
        no_grid: bool = False,
        influence_shading: bool = False,
        uav_observation_radius: bool = False,
        rover_observation_radius: bool = False,
        include_bounds: bool = False,
        downsample: int = 1,
        batch_plot_args: BatchPlotArgs = None
    ):
    """Generate all the joint trajectories in this experiment tree"""

    if out_dir is None:
        # Infer out_dir by replacing 'results' with 'outfigs' in root_dir
        root_str = str(root_dir)
        if 'results' not in root_str:
            raise ValueError("No 'results' folder found in root_dir. out_dir must be specified so plots can be saved somewhere")
        out_dir = Path(root_str.replace('results', 'outfigs'))

    # Get all the directories of joint trajectories
    jt_dirs = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if 'joint_traj.csv' in file:
                jt_dirs.add(Path(root)/file)

    # Sort them
    jt_dirs = [str(jt_dir) for jt_dir in jt_dirs]
    sort_jt_dirs(root_dir, jt_dirs)
    jt_dirs = [Path(jt_dir) for jt_dir in jt_dirs]


    # Plot each one (depending on downsample)
    for jt_dir in jt_dirs[::downsample]:
        dir_list = str(jt_dir).split("/")
        dir_name = "/".join(dir_list[dir_list.index(root_dir.name)+1:-1])
        file_name = jt_dir.name.replace('.csv', '.png')

        plot_joint_trajectory(
            joint_traj_dir=jt_dir,
            num_steps=num_steps,
            individual_colors=individual_colors,
            use_image=use_image,
            no_poi_shading=no_shading,
            no_grid=no_grid,
            influence_shading=influence_shading,
            uav_observation_radius=uav_observation_radius,
            rover_observation_radius=rover_observation_radius,
            include_bounds=include_bounds,
            plot_args=batch_plot_args.build_plot_args(
                title=jt_dir.name,
                output=out_dir/dir_name/file_name
            )
        )

def plot_joint_trajectory_tree(
        root_dir: Path,
        out_dir: Optional[Path] = None,
        num_steps: Optional[int] = None,
        individual_colors: bool = False,
        use_image: bool = False,
        no_poi_shading: bool = False,
        no_grid: bool = False,
        influence_shading: bool = False,
        uav_observation_radius: bool = False,
        rover_observation_radius: bool = False,
        include_bounds: bool = False,
        downsample: int = 1,
        batch_plot_args: BatchPlotArgs = None
    ):
    generate_joint_trajectory_tree_plots(
        root_dir,
        out_dir,
        num_steps,
        individual_colors,
        use_image,
        no_poi_shading,
        no_grid,
        influence_shading,
        uav_observation_radius,
        rover_observation_radius,
        include_bounds,
        downsample,
        batch_plot_args
    )

def generate_config_plot(
        config_dir: Path,
        individual_colors: bool,
        no_shading: bool,
        plot_args: PlotArgs
    ):
    # Load the config
    config = load_config(config_dir)

    # Set up figure
    fig, ax = plot_args.init_figure()

    # plot rovers
    rover_colors = get_rover_colors(individual_colors)
    for i, rover_config in enumerate(config['env']['agents']['rovers']):
        rover_position = rover_config['position']['fixed']
        ax.plot(rover_position[0], rover_position[1], 's', ms=8, color=rover_colors[i%len(rover_colors)])

    # plot uavs
    uav_colors = get_uav_colors(individual_colors)
    for i, uav_config in enumerate(config['env']['agents']['uavs']):
        uav_position = uav_config['position']['fixed']
        ax.plot(uav_position[0], uav_position[1], 'x', ms=8, color=uav_colors[i%len(uav_colors)])

    # plot rover pois (rovers can sense these pois)
    for i, poi_config in enumerate(config['env']['pois']['rover_pois']):
        poi_position = poi_config['position']['fixed']
        plot_poi(ax, poi_config, x=poi_position[0], y=poi_position[1], color='tab:green', radius_shading=not no_shading)

    # plot hidden pois
    for i, poi_config in enumerate(config['env']['pois']['hidden_pois']):
        poi_position = poi_config['position']['fixed']
        plot_poi(ax, poi_config, x=poi_position[0], y=poi_position[1], color='tab:cyan', radius_shading=not no_shading)

    x_bound, y_bound = config['env']['map_size']

    ax.set_xlim([0, x_bound])
    ax.set_ylim([0, y_bound])
    ax.set_aspect('equal')

    plot_args.apply(ax)

    return fig

def plot_config(
        config_dir: Path,
        individual_colors: bool,
        no_shading: bool,
        plot_args: PlotArgs
    ):
    fig = generate_config_plot(config_dir, individual_colors, no_shading, plot_args)
    plot_args.finish_figure(fig)

def generate_learning_curve_tree_plots(
        root_dir: Path,
        out_dir: Path,
        individual_agents: bool,
        batch_plot_args: BatchPlotArgs,
        batch_line_plot_args: BatchLinePlotArgs
    ):
    """Generate all the learning curve plots in this experiment tree"""

    if out_dir is None:
        # Infer out_dir by replacing 'results' with 'outfigs' in root_dir
        root_str = str(root_dir)
        if 'results' not in root_str:
            raise ValueError("No 'results' folder found in root_dir. out_dir must be specified so plots can be saved somewhere")
        out_dir = Path(root_str.replace('results', 'outfigs'))

    fitness_files = set()
    for root, _, files in os.walk(root_dir):
        if 'fitness.csv' in files:
            fitness_files.add(Path(root)/'fitness.csv')

    for fitness_file in fitness_files:
        dir_list = str(fitness_file.parent).split('/')
        dir_name = '/'.join(dir_list[dir_list.index(root_dir.name)+1:])

        file_append = ''
        if individual_agents:
            file_append+='.ind'
        if batch_line_plot_args.window_size is not None:
            file_append+='.w'+str(batch_line_plot_args.window_size)

        plot_learning_curve(
            fitness_dir=fitness_file,
            individual_agents=individual_agents,
            line_plot_args=batch_line_plot_args.build_line_plot_args(),
            plot_args=batch_plot_args.build_plot_args(
                title=dir_name, output=out_dir/dir_name/('learning_curve'+file_append+'.png')
            )
        )

def plot_learning_curve_tree(
        root_dir: Path,
        out_dir: Path,
        individual_agents: bool,
        batch_plot_args: BatchPlotArgs,
        batch_line_plot_args: BatchLinePlotArgs
    ):
    """Plot learning curves for all fitness.csv files in the directory tree"""
    generate_learning_curve_tree_plots(
        root_dir,
        out_dir,
        individual_agents,
        batch_plot_args,
        batch_line_plot_args
    )
