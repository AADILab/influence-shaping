from typing import List, Optional
import os

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from influence.config import load_config

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
        if "rover" in label:
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

def plot_poi(ax, poi_config, x, y, color):
    center_circle = plt.Circle(
        xy = (x, y),
        radius = min(1.0, poi_config['observation_radius']),
        color=color,
        fill=True,
        alpha=0.9
        )
    outer_circle = plt.Circle(
        xy = (x, y),
        radius = poi_config['observation_radius'],
        color=color,
        fill=True,
        alpha=0.1
        )
    ax.add_patch(center_circle)
    ax.add_patch(outer_circle)

def generate_joint_trajectory_plot(joint_traj_dir: Path):
    """Generate plot of the joint trajectory specified in joint_traj_dir"""

    fig, ax = plt.subplots(1,1)

    # Get the joint trajectory
    df = pd.read_csv(joint_traj_dir)

    # Get config for map bounds
    config_dir = joint_traj_dir.parent.parent.parent/'config.yaml'
    config = load_config(config_dir)

    # Get the number of each entity
    num_rovers, num_uavs, num_rover_pois, num_hidden_pois \
        = get_num_entities(labels=df.columns.to_list())

    for i in range(num_rovers):
        ax.plot(df['rover_'+str(i)+'_x'], df['rover_'+str(i)+'_y'], ':', color='tab:purple')
        ax.plot(df['rover_'+str(i)+'_x'].iloc[-1], df['rover_'+str(i)+'_y'].iloc[-1], 's', color='tab:purple')
    for i in range(num_uavs):
        ax.plot(df['uav_'+str(i)+'_x'], df['uav_'+str(i)+'_y'], ':', color='tab:orange')
        ax.plot(df['uav_'+str(i)+'_x'].iloc[-1], df['uav_'+str(i)+'_y'].iloc[-1], 'x', color='tab:orange')
    for i, poi_config in enumerate(config['env']['pois']['rover_pois']):
        plot_poi(ax, poi_config, x=df['rover_poi_'+str(i)+'_x'][0], y=df['rover_poi_'+str(i)+'_y'][0], color='tab:green')
    for i, poi_config in enumerate(config['env']['pois']['hidden_pois']):
        plot_poi(ax, poi_config, x=df['hidden_poi_'+str(i)+'_x'][0], y=df['hidden_poi_'+str(i)+'_y'][0], color='tab:cyan')

    x_bound, y_bound = config['env']['map_size']

    ax.set_xlim([0, x_bound])
    ax.set_ylim([0, y_bound])
    ax.set_aspect('equal')

    return fig

def plot_joint_trajectory(joint_traj_dir: Path, output: Optional[Path], silent: bool):
    fig = generate_joint_trajectory_plot(joint_traj_dir)

    if output is not None:
        if not os.path.exists(output.parent):
            os.makedirs(output.parent)
        fig.savefig(output)
    
    if not silent:
        plt.show()

def generate_learning_curve_plot(fitness_dir, title, individual_agents):
    """Generate plot of the learning curve specified in fitness_dir"""

    fig, ax = plt.subplots(1,1)

    # Get the fitnesses
    df = pd.read_csv(fitness_dir)

    gens = df['generation']
    fits = df['team_fitness_aggregated']

    ax.plot(gens, fits, label='team')

    if individual_agents:
        num_rovers, num_uavs, _, _ = get_num_entities(labels=df.columns.to_list())
        for i in range(num_rovers):
            rover_label = 'rover_'+str(i)+'_'
            fits = df[rover_label]
            ax.plot(gens, fits, label=rover_label)
        for i in range(num_uavs):
            uav_label = 'uav_'+str(i)+'_'
            fits = df[uav_label]
            ax.plot(gens, fits, label=uav_label)
        ax.legend()

    if title:
        ax.set_title(title)

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens.iloc[-1]])
    ax.set_ylim([0, 1.1])

    return fig

def plot_learning_curve(fitness_dir: Path, output: Optional[Path], silent: bool, title: str, individual_agents: str):
    fig = generate_learning_curve_plot(fitness_dir, title, individual_agents)

    if output is not None:
        if not os.path.exists(output.parent):
            os.makedirs(output.parent)
        fig.savefig(output)
    
    if not silent:
        plt.show()

def add_stat_learning_curve(ax: Axes, trials_dir: Path, label: str):
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

    # Plot statistics
    ax.plot(gens, avg, label=label)
    ax.fill_between(gens, low_err, upp_err, alpha=0.2)

    return gens

def generate_stat_learning_curve_plot(trials_dir: Path):
    """Generate plot of statistics of learning given the parent directoy of trials"""

    fig, ax = plt.subplots(1,1)

    gens = add_stat_learning_curve(ax, trials_dir, label=trials_dir.name)

    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.set_xlim([0, gens[-1]])
    ax.set_ylim([0, 1.1])

    return fig

def plot_stat_learning_curve(trials_dir):
    fig = generate_stat_learning_curve_plot(trials_dir)

    plt.show()

def generate_experiment_plot(experiment_dir: Path, title: Optional[str]):
    """Generate plot of experiment using experiment directory
    experiment_dir is parent of parent of trial directories
    """
    fig, ax = plt.subplots(1,1)

    # Get the parent dirs of trials
    dirs = [experiment_dir/dir for dir in os.listdir(experiment_dir)]

    xlim = 0
    for trials_dir in dirs:
        gens = add_stat_learning_curve(ax, trials_dir, label=trials_dir.name)
        if gens[-1] > xlim:
            xlim = gens[-1]
    
    ax.set_xlabel('Generations')
    ax.set_ylabel('Performance')

    ax.legend()

    ax.set_xlim([0, gens[-1]])
    ax.set_ylim([0, 1.1])

    if title:
        ax.set_title(title)

    return fig

def plot_experiment(experiment_dir: Path, output: Optional[Path], silent: bool, title: str):
    fig = generate_experiment_plot(experiment_dir, title)

    if output is not None:
        if not os.path.exists(output.parent):
            os.makedirs(output.parent)
        fig.savefig(output)
    
    if not silent:
        plt.show()
