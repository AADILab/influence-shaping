from typing import List, Optional
import os

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from influence.config import load_config
from influence.parsing import PlotArgs, LinePlotArgs, BatchPlotArgs, BatchLinePlotArgs

def get_num_entities(labels: List[str]):
    num_rovers = 0
    num_uavs = 0
    num_hidden_pois = 0
    num_rover_pois = 0

    def get_agent_id(label):
        return int(label.split("_")[1])
    
    def get_poi_id(label):
        return int(label.split("_")[2])

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
        radius = min(1.0, poi_config['observation_radius']),
        color=color,
        fill=True,
        alpha=0.9
        )
    ax.add_patch(center_circle)
    if radius_shading:
        outer_circle = plt.Circle(
            xy = (x, y),
            radius = poi_config['observation_radius'],
            color=color,
            fill=True,
            alpha=0.05
            )
        ax.add_patch(outer_circle)

def get_rover_colors(individual_colors: bool):
    if individual_colors:
        rover_colors = plt.cm.Set1.colors[:1]+plt.cm.Set1.colors[3:]
    else:
        rover_colors = ['tab:purple']
    return rover_colors

def get_uav_colors(individual_colors: bool):
    if individual_colors:
        uav_colors = plt.cm.Dark2.colors[1:]
    else:
        uav_colors = ['tab:orange']
    return uav_colors

def add_rover_trajectories(ax: Axes, df: pd.DataFrame, num_rovers: int, individual_colors: bool):
    # if individual_colors:
    #     rover_colors = plt.cm.Set1.colors[:1]+plt.cm.Set1.colors[3:]
    # else:
    #     rover_colors = ['tab:purple']*num_rovers
    rover_colors = get_rover_colors(individual_colors)
    for i in range(num_rovers):
        ax.plot(df['rover_'+str(i)+'_x'], df['rover_'+str(i)+'_y'], ':', lw=2, color=rover_colors[i%len(rover_colors)])
        ax.plot(df['rover_'+str(i)+'_x'].iloc[-1], df['rover_'+str(i)+'_y'].iloc[-1], 's', ms=8, color=rover_colors[i%len(rover_colors)])

def add_uav_trajectories(ax: Axes, df: pd.DataFrame, num_uavs: int, individual_colors: bool):
    uav_colors = get_uav_colors(individual_colors)
    for i in range(num_uavs):
        ax.plot(df['uav_'+str(i)+'_x'], df['uav_'+str(i)+'_y'], ':', lw=2, color=uav_colors[i%len(uav_colors)])
        ax.plot(df['uav_'+str(i)+'_x'].iloc[-1], df['uav_'+str(i)+'_y'].iloc[-1], 'x', ms=8, color=uav_colors[i%len(uav_colors)])

def generate_joint_trajectory_plot(joint_traj_dir: Path, individual_colors: bool, no_shading: bool, plot_args: PlotArgs):
    """Generate plot of the joint trajectory specified in joint_traj_dir"""

    fig, ax = plt.subplots(1,1)

    # Get the joint trajectory
    df = pd.read_csv(joint_traj_dir)

    # Get config for map bounds
    config_dir = joint_traj_dir.parent.parent.parent/'config.yaml'
    config = load_config(config_dir)

    # Get the number of each entity
    num_rovers, num_uavs, _, _ \
        = get_num_entities(labels=df.columns.to_list())

    add_rover_trajectories(ax, df, num_rovers, individual_colors)
    add_uav_trajectories(ax, df, num_uavs, individual_colors)
    for i, poi_config in enumerate(config['env']['pois']['rover_pois']):
        plot_poi(ax, poi_config, x=df['rover_poi_'+str(i)+'_x'][0], y=df['rover_poi_'+str(i)+'_y'][0], color='tab:green', radius_shading=not no_shading)
    for i, poi_config in enumerate(config['env']['pois']['hidden_pois']):
        plot_poi(ax, poi_config, x=df['hidden_poi_'+str(i)+'_x'][0], y=df['hidden_poi_'+str(i)+'_y'][0], color='tab:cyan', radius_shading=not no_shading)

    x_bound, y_bound = config['env']['map_size']

    ax.set_xlim([0, x_bound])
    ax.set_ylim([0, y_bound])
    ax.set_aspect('equal')

    plot_args.apply(ax)

    return fig

def plot_joint_trajectory(joint_traj_dir: Path, individual_colors: bool, no_shading: bool, plot_args: PlotArgs):
    fig = generate_joint_trajectory_plot(joint_traj_dir, individual_colors, no_shading, plot_args)
    plot_args.finish_figure(fig)

def add_learning_curve(ax: Axes, df: pd.DataFrame, line_plot_args: LinePlotArgs, label: str = 'team'):
    """Add the team's learning curve from the specified fitness directory to the Axes object"""

    # Get the points for plotting team fitness
    gens, fits = line_plot_args.get_pts(xs=df['generation'], ys=df['team_fitness_aggregated'])
    ax.plot(gens, fits, label=label)

    return gens

def generate_learning_curve_plot(fitness_dir, individual_agents, line_plot_args: LinePlotArgs, plot_args: PlotArgs):
    """Generate plot of the learning curve specified in fitness_dir"""

    fig, ax = plt.subplots(1,1)

    # Get the fitnesses
    df = pd.read_csv(fitness_dir)

    # Get points for plotting team fitness
    gens = add_learning_curve(ax, df, line_plot_args)

    if individual_agents:
        num_rovers, num_uavs, _, _ = get_num_entities(labels=df.columns.to_list())
        for i in range(num_rovers):
            rover_label = 'rover_'+str(i)+'_'
            fits = line_plot_args.get_ys(ys=df[rover_label])
            ax.plot(gens, fits, label=rover_label)
        for i in range(num_uavs):
            uav_label = 'uav_'+str(i)+'_'
            fits = line_plot_args.get_ys(ys=df[uav_label])
            ax.plot(gens, fits, label=uav_label)
        ax.legend()

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens.iloc[-1]])
    ax.set_ylim([0, 1.1])

    plot_args.apply(ax)

    return fig

def plot_learning_curve(fitness_dir: Path, individual_agents: str, line_plot_args: LinePlotArgs, plot_args: PlotArgs):
    fig = generate_learning_curve_plot(fitness_dir, individual_agents, line_plot_args, plot_args)
    plot_args.finish_figure(fig)

def add_stat_learning_curve(ax: Axes, trials_dir: Path, label: str, line_plot_args: LinePlotArgs):
    # Get the directories of trials
    dirs = [trials_dir/dir for dir in os.listdir(trials_dir) if 'trial_' in dir]

    # Get the fitnesses in each trial
    dfs = [pd.read_csv(dir/'fitness.csv') for dir in dirs]

    # Figure out which trial ran the shortest 
    # (We can only accurately compute statistics for generations that we have all trials' output for)
    ind = min([len(df['team_fitness_aggregated']) for df in dfs])

    # Compute the statistics
    avg = np.average([df['team_fitness_aggregated'][:ind] for df in dfs], axis=0)
    err = np.std([df['team_fitness_aggregated'][:ind] for df in dfs], axis=0) / np.sqrt(len(dfs))
    upp_err = avg+err
    low_err = avg-err
    gens = list(range(len(avg)))

    # Clean up data
    gens, avg = line_plot_args.get_pts(gens, avg)
    low_err = line_plot_args.get_ys(low_err)
    upp_err = line_plot_args.get_ys(upp_err)

    # Plot statistics
    ax.plot(gens, avg, label=label)
    ax.fill_between(gens, low_err, upp_err, alpha=0.2)

    return gens

def generate_stat_learning_curve_plot(trials_dir: Path, individual_trials: bool, line_plot_args: LinePlotArgs, plot_args: PlotArgs):
    """Generate plot of statistics of learning given the parent directoy of trials"""

    fig, ax = plt.subplots(1,1)

    # Plot each trial individual if specified
    if individual_trials:
        # Get the directories of trials
        dirs = [trials_dir/dir for dir in os.listdir(trials_dir) if 'trial_' in dir]

        # Sort dirs
        dirs.sort(key=lambda x: int(str(x).split('_')[-1]))

        # Get the fitnesses in each trial
        dfs = [pd.read_csv(dir/'fitness.csv') for dir in dirs]

        # Plot each trial's fitness throughout training
        for df, dir in zip(dfs, dirs):
            gens = add_learning_curve(ax, df, line_plot_args, label=dir.name)

        # Put gens in the expected format
        gens = np.array(gens)

        ax.legend()
    else:
        gens = add_stat_learning_curve(ax, trials_dir, individual_trials, label=trials_dir.name, line_plot_args=line_plot_args)

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens[-1]])
    ax.set_ylim([0, 1.1])

    plot_args.apply(ax)

    return fig

def plot_stat_learning_curve(trials_dir, individual_trials, line_plot_args, plot_args):
    fig = generate_stat_learning_curve_plot(trials_dir, individual_trials, line_plot_args, plot_args)
    plot_args.finish_figure(fig)

def generate_comparison_plot(experiment_dir: Path, line_plot_args: LinePlotArgs, plot_args: PlotArgs):
    """Generate plot of experiment using experiment directory
    experiment_dir is parent of parent of trial directories
    """
    fig, ax = plt.subplots(1,1)

    # Get the parent dirs of trials
    dirs = [experiment_dir/dir for dir in os.listdir(experiment_dir)]

    xlim = 0
    for trials_dir in dirs:
        gens = add_stat_learning_curve(ax, trials_dir, label=trials_dir.name, line_plot_args=line_plot_args)
        if gens[-1] > xlim:
            xlim = gens[-1]
    
    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.legend()

    ax.set_xlim([0, gens[-1]])

    plot_args.apply(ax)

    return fig

def plot_comparison(experiment_dir: Path, line_plot_args: LinePlotArgs, plot_args: PlotArgs):
    fig = generate_comparison_plot(experiment_dir, line_plot_args, plot_args)
    plot_args.finish_figure(fig)

def get_example_trial_dirs(parent_dir: Path):
    dirs = [parent_dir/dir for dir in os.list(parent_dir) if 'trial_' in dir]
    dfs = [pd.read_csv(dir/'fitness.csv') for dir in dirs]
    fits = [df['team_fitness_aggregated'] for df in dfs]
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

def generate_experiment_tree_plots(root_dir: Path, out_dir: Path, batch_plot_args: BatchPlotArgs, batch_line_plot_args: BatchLinePlotArgs):
    """Generate all the plots in this experiment tree"""
    
    experiment_dirs = set()
    trial_parent_dirs = set()
    for root, _, files in os.walk(root_dir):
        if 'config.yaml' in files:
            experiment_dirs.add(Path(root).parent)
            trial_parent_dirs.add(Path(root))

    for dir_ in experiment_dirs:
        dir_list = str(dir_).split("/")
        dir_name = "/".join(dir_list[dir_list.index(root_dir.name)+1:])

        plot_comparison(
            experiment_dir=dir_,
            line_plot_args=batch_line_plot_args.build_line_plot_args(),
            plot_args=batch_plot_args.build_plot_args(
                title=dir_name, output=out_dir/dir_name/'comparison.png'
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

def generate_joint_trajectory_tree_plots(root_dir: Path, out_dir: Path, individual_colors: bool, no_shading: bool, downsample: int, batch_plot_args: BatchPlotArgs):
    """Generate all the joint trajectories in this experiment tree"""

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
    

    # Plot each one
    for jt_dir in jt_dirs:
        dir_list = str(jt_dir).split("/")
        dir_name = "/".join(dir_list[dir_list.index(root_dir.name)+1:-1])
        file_name = jt_dir.name.replace('.csv', '.png')

        plot_joint_trajectory(
            joint_traj_dir=jt_dir,
            individual_colors=individual_colors,
            no_shading=no_shading,
            plot_args=batch_plot_args.build_plot_args(
                title=jt_dir.name, 
                output=out_dir/dir_name/file_name
            )
        )

def plot_comparison_tree(root_dir: Path, out_dir: Path, batch_plot_args: BatchPlotArgs, batch_line_plot_args: BatchLinePlotArgs):
    generate_experiment_tree_plots(root_dir, out_dir, batch_plot_args, batch_line_plot_args)

def plot_joint_trajectory_tree(root_dir: Path, out_dir: Path, individual_colors: bool, no_shading: bool, downsample: int, batch_plot_args: BatchPlotArgs):
    generate_joint_trajectory_tree_plots(root_dir, out_dir, individual_colors, no_shading, downsample, batch_plot_args)

def generate_config_plot(config_dir: Path, individual_colors: bool, no_shading: bool, plot_args: PlotArgs):
    # Load the config
    config = load_config(config_dir)

    # Set up figure
    fig, ax = plt.subplots(1,1)

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

def plot_config(config_dir: Path, individual_colors: bool, no_shading: bool, plot_args: PlotArgs):
    fig = generate_config_plot(config_dir, individual_colors, no_shading, plot_args)
    plot_args.finish_figure(fig)
