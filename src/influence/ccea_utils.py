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
    def __init__(self, fitnesses, joint_trajectory):
        self.fitnesses = fitnesses
        self.joint_trajectory = joint_trajectory

class RolloutPackIn():
    def __init__(self, policies, seed):
        self.policies = policies
        self.seed = seed
        self.fitness = None
        self.agg_fitness = None

class Individual:
    def __init__(self, weights, temp_id):
        self.weights = weights
        self.temp_id = temp_id
        self.rollout_team_fitnesses = []
        self.rollout_shaped_fitnesses = []
        self.team_fitness = None
        self.shaped_fitness = None
