from typing import List
import numpy as np
import random

def bound_value(value, upper, lower):
    """Bound the value between an upper and lower bound"""
    if value > upper:
        return upper
    elif value < lower:
        return lower
    else:
        return value

def bound_velocity(velocity, max_velocity):
    """Bound the velocity to meet max velocity constraint"""
    if velocity > max_velocity:
        return max_velocity
    elif velocity < -max_velocity:
        return -max_velocity
    else:
        return velocity

def bound_velocity_arr(velocity_arr, max_velocity):
    """Bound the velocities in a 1D array to meet the max velocity constraint"""
    return np.array([bound_velocity(velocity, max_velocity) for velocity in velocity_arr])

def getRandomWeights(num_weights: int) -> List[float]:
    return [2*random.random()-1 for _ in range(num_weights)]

class FollowPolicy():
    @staticmethod
    def forward(observation: np.ndarray) -> int:
        if all(observation==-1):
            return [0.0 for _ in observation]+[1]
        else:
            f_obs = [o if o !=-1 else np.inf for o in observation]
            action = [0.0 for _ in observation]
            action[np.argmin(f_obs)] = 1
            return action

class JointTrajectory():
    def __init__(self, joint_state_trajectory, joint_observation_trajectory, joint_action_trajectory):
        self.states = joint_state_trajectory
        self.observations = joint_observation_trajectory
        self.actions = joint_action_trajectory

class RolloutPackOut():
    def __init__(self, shaped_fitnesses, team_fitness, joint_trajectory):
        self.shaped_fitnesses = shaped_fitnesses
        self.team_fitness = team_fitness
        self.joint_trajectory = joint_trajectory

class RolloutPackIn():
    def __init__(self, team, seed):
        self.team = team
        self.seed = seed

class RolloutPack():
    def __init__(self, team, seed, shaped_fitnesses, team_fitness, joint_trajectory):
        self.team = team
        self.seed = seed
        self.shaped_fitnesses = shaped_fitnesses
        self.team_fitness = team_fitness
        self.joint_trajectory = joint_trajectory

class Individual:
    def __init__(self, weights, temp_id):
        self.weights = weights
        self.temp_id = temp_id
        self.rollout_team_fitnesses = []
        self.rollout_shaped_fitnesses = []
        self.team_fitness = None
        self.shaped_fitness = None

class TeamPack():
    def __init__(self, team: List[Individual], rollout_packs: List[RolloutPack]):
        self.team = team
        self.rollout_packs = rollout_packs

        self.collapsed_team_fitness = collapse_team_fitness(rollout_packs)
        self.collapsed_shaped_fitness = collapse_shaped_fitnesses(rollout_packs)

class Checkpoint():
    def __init__(self, loaded: bool, agent_pops=None, team_pop=None, team_packs=None, gen=None):
        self.loaded = loaded
        self.agent_pops = agent_pops
        self.team_pop = team_pop
        self.team_packs = team_packs
        self.gen = gen

IndividualPopulation = List[Individual]
Team = List[Individual]
TeamPopulation = List[Team]

def build_rollout_packs(rollout_pack_ins: List[RolloutPackIn], rollout_pack_outs: List[RolloutPackOut])->List[RolloutPack]:
    rollout_packs = []
    for in_, out in zip(rollout_pack_ins, rollout_pack_outs):
        rollout_packs.append(
            RolloutPack(
                in_.team,
                in_.seed,
                out.shaped_fitnesses,
                out.team_fitness,
                out.joint_trajectory
            )
        )
    return rollout_packs

def collapse_team_fitness(rollout_packs: List[RolloutPack]):
    return sum(rp.team_fitness for rp in rollout_packs) / len(rollout_packs)

def collapse_shaped_fitnesses(rollout_packs: List[RolloutPack]):
    fit_lists = [r.shaped_fitnesses for r in rollout_packs]
    return [sum(agent_values) / len(agent_values) for agent_values in zip(*fit_lists)]

def build_team_pack(rollout_packs: List[RolloutPack])->TeamPack:
    return TeamPack(
        team=rollout_packs[0].team,
        rollout_packs=rollout_packs
    )

def build_team_packs(rollout_packs: List[RolloutPack], num_rollouts_per_team: int) -> List[TeamPack]:
    rpt = num_rollouts_per_team
    rollouts_packs_split_by_team = [
        rollout_packs[i:i + rpt] for i in range(0, len(rollout_packs), rpt)
    ]
    team_packs = [
        build_team_pack(rps) for rps in rollouts_packs_split_by_team
    ]
    return team_packs
