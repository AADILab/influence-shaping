from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm

import multiprocessing
import random
import pickle
from typing import List, Union, Optional

from influence.evo_network import NeuralNetwork

from influence.librovers import rovers
from influence.custom_env import createEnv
from copy import deepcopy
import numpy as np
import random
import os
from pathlib import Path
import yaml
import pprint
from tqdm import tqdm
import pandas as pd

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

# class CustomIndividual(creator.Individual):
#     def __init__(self):
#         super().__init__()
#         self.team_fitness = None

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

# class JointTrajectory():
#     def __init__(self, joint_state_trajectory, joint_observation_trajectory, joint_action_trajectory):
#         self.states = joint_state_trajectory
#         self.observations = joint_observation_trajectory
#         self.actions = joint_action_trajectory

#     def to_df(self):
#         pass

class EvalOut():
    def __init__(self, fitnesses, joint_trajectory):
        self.fitnesses = fitnesses
        self.joint_trajectory = joint_trajectory

class TeamEvalIn():
    def __init__(self, individuals, seed):
        # Each policy is an "individual"
        # Seed is for evaluation in random environments
        self.individuals = individuals
        self.seed = seed

class TeamSummary():
    def __init__(self, individuals, seeds, eval_outs):
        self.individuals = individuals
        self.seeds = seeds
        self.eval_outs = eval_outs

class CooperativeCoevolutionaryAlgorithm():
    def __init__(self, config_dir):
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = self.config_dir.parent

        with open(str(self.config_dir), 'r') as file:
            self.config = yaml.safe_load(file)

        # Start by setting up variables for different agents
        self.num_rovers = len(self.config["env"]["agents"]["rovers"])
        self.num_uavs = len(self.config["env"]["agents"]["uavs"])
        self.subpopulation_size = self.config["ccea"]["population"]["subpopulation_size"]
        self.num_hidden = self.config["ccea"]["network"]["hidden_layers"]

        self.num_rover_pois = len(self.config["env"]["pois"]["rover_pois"])
        self.num_hidden_pois = len(self.config["env"]["pois"]["hidden_pois"])

        self.selection_mechanism = self.config['ccea']['selection']['mechanism']
        self.sort_teams_by_sum_agent_fitness = False
        if self.selection_mechanism == 'mixed_n_elites_binary_tournament':
            self.n_preserve_team_elites = self.config['ccea']['selection']['mixed_n_elites_binary_tournament']['n_preserve_team_elites']
            self.n_preserve_individual_elites = self.config['ccea']['selection']['mixed_n_elites_binary_tournament']['n_preserve_individual_elites']
            self.n_team_elites = self.config['ccea']['selection']['mixed_n_elites_binary_tournament']['n_team_elites']
            self.n_individual_elites = self.config['ccea']['selection']['mixed_n_elites_binary_tournament']['n_individual_elites']

            self.n_preserve_elites = self.n_preserve_team_elites + self.n_preserve_individual_elites
            self.n_current_elites = self.n_team_elites + self.n_individual_elites

            self.total_elites = self.n_preserve_elites + self.n_current_elites

            self.num_mutants = self.subpopulation_size - self.total_elites
            if 'sort_teams_by_sum_agent_fitness' in self.config['ccea']['selection']['mixed_n_elites_binary_tournament']:
                self.sort_teams_by_sum_agent_fitness = self.config['ccea']['selection']['mixed_n_elites_binary_tournament']['sort_teams_by_sum_agent_fitness']

        elif self.selection_mechanism == 'epsilon_greedy':
            self.num_mutants = int(self.subpopulation_size / 2)
            self.num_elites = self.subpopulation_size - self.num_mutants # Just in case subpopulation size is an odd number
            self.total_elites = self.num_elites
            self.epsilon = self.config['ccea']['selection']['epsilon_greedy']['epsilon']
            self.n_preserve_elites = 0

        # Evaluation settings
        self.num_rollouts_per_team = self.config['ccea']['evaluation']['multi_evaluation']['num_rollouts_per_team']
        self.agg_across_rollouts = self.config['ccea']['evaluation']['multi_evaluation']['aggregation_across_rollouts']
        self.num_teams_per_evaluation = self.config['ccea']['evaluation']['multi_evaluation']['num_teams_per_evaluation']
        self.agg_across_teams = self.config['ccea']['evaluation']['multi_evaluation']['aggregation_across_teams']
        self.choose_team_aggregation_across_rollouts = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['choose_team_aggregation_across_rollouts']
        self.choose_shaped_aggregation_across_rollouts = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['choose_shaped_aggregation_across_rollouts']
        self.test_aggregation_across_rollouts = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['test_aggregation_across_rollouts']
        self.test_num_rollouts = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['test_num_rollouts']
        self.resim_test_evaluation = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['switch']
            # Whether or not we even resimulate for testing or just use the training rollouts
        self.num_team_champions = \
            self.config['ccea']['evaluation']['resim_test_evaluation']['num_team_champions']
        self.num_gens_between_save_champions = \
            self.config['data']['save_champion_trajectories']['num_gens_between_save']

        if 'save_elite_fitness' not in self.config['data']:
            self.config['data']['save_elite_fitness'] = {}
        if 'switch' not in self.config['data']['save_elite_fitness']:
            self.config['data']['save_elite_fitness']['switch'] = False

        self.num_steps = self.config["ccea"]["num_steps"]

        self.template_policies = self.get_template_policies('rovers')+self.get_template_policies('uavs')
        # self.template_nns = self.get_template_nns('rovers')+self.get_template_nns('uavs')
        self.nn_sizes = [template_nn.num_weights if type(template_nn) is NeuralNetwork else None for template_nn in self.template_policies]

        # Make sure each agent has a sensor type config set
        for agent_config in self.config['env']['agents']['rovers']+self.config['env']['agents']['uavs']:
            if 'sensor' not in agent_config:
                agent_config['sensor'] = {'type': 'SmartLidar'}
            elif 'type' not in agent_config['sensor']:
                agent_config['sensor']['type'] = 'SmartLidar'

        # Save number of sensors that each agent has
        self.num_sensors_rovers = []
        for i in range(self.num_rovers):
            if self.config["env"]["agents"]["rovers"][i]['sensor']['type'] == 'SmartLidar':
                self.num_sensors_rovers.append(3*int(360/self.config["env"]["agents"]["rovers"][i]["resolution"]))
            elif self.config['env']['agents']['rovers'][i]['sensor']['type'] == 'UavDistanceLidar':
                self.num_sensors_rovers.append(self.num_uavs)
        self.num_sensors_uavs = []
        for i in range(self.num_uavs):
            if self.config["env"]["agents"]["uavs"][i]['sensor']['type'] == 'SmartLidar':
                self.num_sensors_uavs.append(3*int(360/self.config["env"]["agents"]["uavs"][i]["resolution"]))
            elif self.config['env']['agents']['uavs'][i]['sensor']['type'] == 'UavDistanceLidar':
                self.num_sensors_uavs.append(self.num_uavs)

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data saving variables
        self.save_champion_trajectories = self.config["data"]["save_champion_trajectories"]["switch"]
        self.num_gens_between_save_eval_traj = self.config["data"]["save_champion_trajectories"]["num_gens_between_save"]
        if 'checkpoints' in self.config['data'] and 'save' in self.config['data']['checkpoints']:
            self.save_checkpoint = self.config["data"]["checkpoints"]["save"]
        else:
            self.save_checkpoint = False
        if 'checkpoints' in self.config['data'] and 'frequency' in self.config['data']['checkpoints']:
            self.num_gens_between_checkpoint = self.config["data"]["checkpoints"]["frequency"]
        else:
            self.num_gens_between_checkpoint = 0
        if 'checkpoints' in self.config['data'] and 'delete_previous' in self.config['data']['checkpoints']:
            self.delete_previous_checkpoint = self.config["data"]["checkpoints"]["delete_previous"]
        else:
            self.delete_previous_checkpoint = True

        # Check if we are using a random seed
        self.random_seed_val = None
        if 'debug' in self.config:
            if 'random_seed' in self.config['debug']:
                if 'set_seed' in self.config['debug']['random_seed']:
                    self.random_seed_val = self.config['debug']['random_seed']['set_seed']

        # Check if we are incrementing that seed
        self.increment_seed_every_trial = False
        if 'debug' in self.config:
            if 'random_seed' in self.config['debug']:
                if 'increment_every_trial' in self.config['debug']['random_seed']:
                    self.increment_seed_every_trial = self.config['debug']['random_seed']['increment_every_trial']

        # Create the type of fitness we're optimizing
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()
        if self.use_multiprocessing:
            self.pool = multiprocessing.Pool(processes=self.num_threads)
            self.map = self.pool.map_async
        else:
            self.toolbox.register("map", map)
            self.map = map

    def reset_seed(self):
        """Resets the random seed to what was specified in config"""
        self.random_seed_val = self.config['debug']['random_seed']['set_seed']

    def increment_seed(self, i: int):
        """Manually increment the seed"""
        self.random_seed_val += i

    def get_seed(self):
        if self.random_seed_val is None:
            return None
        else:
            out = self.random_seed_val
            self.random_seed_val += 1
            return out

    # This makes it possible to pass evaluation to multiprocessing
    # Without this, the pool tries to pickle the entire object, including itself
    # which it cannot do
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def get_num_sensors(agent_config: dict, num_uavs: int):
        if agent_config['sensor']['type'] == 'SmartLidar':
            num_inputs = 3*int(360/agent_config["resolution"])
        elif agent_config['sensor']['type'] == 'UavDistanceLidar':
            num_inputs = num_uavs
        return num_inputs

    def get_template_nn(self, agent_config: dict):
        # Figure out number of inputs to nn from observation size
        num_inputs = self.get_num_sensors(agent_config, self.num_uavs)
        # Figure out number of outputs from action
        if agent_config['action']['type'] == 'dxdy':
            num_outputs = 2
            output_activation_function='tanh'
        elif agent_config['action']['type'] == 'pick_uav':
            num_outputs = self.num_uavs+1
            output_activation_function='softmax'
        # Create template nn
        return NeuralNetwork(
            num_inputs=num_inputs,
            num_hidden=self.num_hidden,
            num_outputs=num_outputs,
            hidden_activation_func='tanh',
            output_activation_func=output_activation_function
        )

    def get_template_policies(self, agent_type: str):
        template_policies = []
        for i in range(len(self.config['env']['agents'][agent_type])):
            # Fill defaults for sensor, action, policy
            if 'sensor' not in self.config['env']['agents'][agent_type][i]:
                self.config['env']['agents'][agent_type][i]['sensor'] = {'type' : 'SmartLidar'}
            elif 'type' not in self.config['env']['agents'][agent_type][i]['sensor']:
                self.config['env']['agents'][agent_type][i]['sensor']['type'] = 'SmartLidar'
            if 'action' not in self.config['env']['agents'][agent_type][i]:
                self.config['env']['agents'][agent_type][i]['action'] = {'type': 'dxdy'}
            elif 'type' not in self.config['env']['agents'][agent_type][i]['action']:
                self.config['env']['agents'][agent_type][i]['action']['type'] = 'dxdy'
            if 'policy' not in self.config['env']['agents'][agent_type][i]:
                self.config['env']['agents'][agent_type][i]['policy'] = {'type': 'network'}
            elif 'type' not in self.config['env']['agents'][agent_type][i]['policy']:
                self.config['env']['agents'][agent_type][i]['policy']['type'] = 'network'
            # Now figure out what template policy this agent uses
            if self.config['env']['agents'][agent_type][i]['policy']['type'] == 'network':
                template_policy = self.get_template_nn(self.config['env']['agents'][agent_type][i])
            elif self.config['env']['agents'][agent_type][i]['policy']['type'] == 'follow':
                template_policy = FollowPolicy()
            template_policies.append(template_policy)
        return template_policies

    def generateTemplateNN(self, num_sectors):
        agent_nn = NeuralNetwork(num_inputs=3*num_sectors, num_hidden=self.num_hidden, num_outputs=2)
        return agent_nn

    def generateWeight(self):
        return random.uniform(self.config["ccea"]["weight_initialization"]["lower_bound"], self.config["ccea"]["weight_initialization"]["upper_bound"])

    def generateIndividual(self, individual_size, _id):
        individual = tools.initRepeat(creator.Individual, self.generateWeight, n=individual_size)
        individual._id = _id
        individual.shaped_fitnesses = []
        individual.team_fitnesses = []
        return individual

    def generateRoverIndividual(self):
        return self.generateIndividual(individual_size=self.rover_nn_size)

    def generateUAVIndividual(self):
        return self.generateIndividual(individual_size=self.uav_nn_size)

    def generateRoverSubpopulation(self):
        return tools.initRepeat(list, self.generateRoverIndividual, n=self.config["ccea"]["population"]["subpopulation_size"])

    def generateUAVSubpopulation(self):
        return tools.initRepeat(list, self.generateUAVIndividual, n=self.config["ccea"]["population"]["subpopulation_size"])

    def populations(self):
        pop = []
        # Generating subpopulation for each agent
        for agent_id in range(self.num_rovers+self.num_uavs):
            if type(self.template_policies[agent_id]) is NeuralNetwork:
                # Filling subpopulation for each agent
                subpop=[]
                for _id in range(self.config["ccea"]["population"]["subpopulation_size"]):
                    subpop.append(self.generateIndividual(individual_size=self.nn_sizes[agent_id], _id=_id))
            else:
                # Subpopulation of None for fixed policies
                subpop=[]
                for _ in range(self.config["ccea"]["population"]["subpopulation_size"]):
                    subpop.append(None)
            pop.append(subpop)
        return pop

    def formEvaluationTeam(self, population):
        policies = []
        for subpop in population:
            if subpop[0] is None:
                best_ind = 0
            else:
                # Use max with a key function to get the individual with the highest fitness[0] value
                best_ind = max(subpop, key=lambda ind: ind.fitness.values[0])
            policies.append(best_ind)
        return TeamEvalIn(policies, self.get_seed())

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team _ times
        eval_teams = [self.formEvaluationTeam(population) for _ in range(self.num_rollouts_per_team)]
        # Evaluate the teams
        return self.evaluateTeams(eval_teams)

    def formTeams(self, populations, skip_preserved=True) -> List:
        # Base case: There are only preserved elites left
        # For each population
        #   Pick a random policy from the population, excluding preserved elites
        #   Put that policy onto the team
        # Return the team, and the downsized populations with those policies removed

        # Base Case. Only preserved elites are left.
        if len(populations[0]) <= self.n_preserve_elites and skip_preserved:
            return []

        # Base Case. No individuals left
        elif len(populations[0]) <= 0:
            return []

        # Standard Case. Form a team and keep going!
        team = []
        reduced_populations = []
        for pop in populations:
            if skip_preserved:
                # Protect preserved elites
                index = int(random.choice(self.n_preserve_elites+np.arange(len(pop)-self.n_preserve_elites)))
            else:
                # Pick anyone!
                index = int(random.choice(np.arange(len(pop))))
            team.append(pop[index])
            reduced_populations.append(pop[:index]+pop[index+1:])
        return [team]+self.formTeams(reduced_populations, skip_preserved=skip_preserved)

    def buildMap(self, teams):
        return self.map(self.evaluateTeam, teams)

    def evaluateTeams(self, teams: List[TeamEvalIn]):
        if self.use_multiprocessing:
            jobs = self.buildMap(teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.buildMap(teams))
        return eval_infos

    def evaluateTeam(self, team: TeamEvalIn, compute_team_fitness=True):
        return self.evaluateTeamStatic(
            team,
            self.template_policies,
            self.config,
            self.num_rovers,
            self.num_uavs,
            self.num_steps,
            compute_team_fitness
        )

    @staticmethod
    def evaluateTeamStatic(
        team: TeamEvalIn,
        template_policies: List[Union[NeuralNetwork|FollowPolicy]],
        config: dict,
        num_rovers: int,
        num_uavs: int,
        num_steps: int,
        compute_team_fitness=True
    ):
        # Set up random seed if one was specified
        if not isinstance(team, TeamEvalIn):
            raise Exception('team should be TeamEvalIn')
        if team.seed is not None:
            random.seed(team.seed)
            np.random.seed(team.seed)

        # Set up networks
        # TODO: Make it so that each agent can have a different size NN?
        #       (This lets us give rovers one size and uavs a different size when we change their
        #        observations and actions)
        # agent_nns = [deepcopy(template_nn) for template_nn in self.template_nns]
        agent_policies = [deepcopy(template_policy) for template_policy in template_policies]

        # Load in the weights
        for nn, individual in zip(agent_policies, team.individuals):
            if type(nn) is NeuralNetwork:
                nn.setWeights(individual)

        # Set up the enviornment
        env = createEnv(config)

        observations, _ = env.reset()

        # For saving rollout information
        num_rover_pois = len(config['env']['pois']['rover_pois'])
        joint_trajectory = {}
        for i, agent in enumerate(env.rovers()[:num_rovers]):
            joint_trajectory['rover_'+str(i)+'_x'] = [agent.position().x]
            joint_trajectory['rover_'+str(i)+'_y'] = [agent.position().y]
            joint_trajectory['rover_'+str(i)+'_dx'] = []
            joint_trajectory['rover_'+str(i)+'_dy'] = []
            observation = observations[i]
            for j in range(len(observation)):
                joint_trajectory['rover_'+str(i)+'_obs_'+str(j)] = [observation(j)]
        for i, agent in enumerate(env.rovers()[num_rovers:]):
            joint_trajectory['uav_'+str(i)+'_x'] = [agent.position().y]
            joint_trajectory['uav_'+str(i)+'_y'] = [agent.position().y]
            joint_trajectory['uav_'+str(i)+'_dx'] = []
            joint_trajectory['uav_'+str(i)+'_dy'] = []
            observation = observations[num_rovers+i]
            for j in range(len(observation)):
                joint_trajectory['uav_'+str(i)+'_obs_'+str(j)] = [observation(j)]
        for i, poi in enumerate(env.pois()[:num_rover_pois]):
            joint_trajectory['rover_poi_'+str(i)+'_x'] = [poi.position().x]
            joint_trajectory['rover_poi_'+str(i)+'_y'] = [poi.position().y]
        for i, poi in enumerate(env.pois()[num_rover_pois:]):
            joint_trajectory['hidden_poi_'+str(i)+'_x'] = [poi.position().x]
            joint_trajectory['hidden_poi_'+str(i)+'_y'] = [poi.position().y]


        # agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
        # poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

        # observations_arrs = []
        # for observation in observations:
        #     observation_arr = []
        #     for i in range(len(observation)):
        #         observation_arr.append(observation(i))
        #     observation_arr = np.array(observation_arr, dtype=np.float64)
        #     observations_arrs.append(observation_arr)

        # joint_state_trajectory = [agent_positions+poi_positions]
        # joint_observation_trajectory = [observations_arrs]
        # joint_action_trajectory = []

        for _ in range(num_steps):
            # Compute the actions of all rovers
            # observation_arrs = []
            actions_arrs = []
            actions = []
            for ind, (observation, agent_nn) in enumerate(zip(observations, agent_policies)):
                # observation_arr = []
                # for i in range(len(observation)):
                #     observation_arr.append(observation(i))
                # observation_arr = np.array(observation_arr, dtype=np.float64)
                slist = str(observation.transpose()).split(" ")
                flist = list(filter(None, slist))
                nlist = [float(s) for s in flist]
                observation_arr = np.array(nlist, dtype=np.float64)
                # print("observation_arr:", observation_arr)
                action_arr = agent_nn.forward(observation_arr)
                if (config['env']['agents']['rovers']+config['env']['agents']['uavs'])[ind]['action']['type'] == 'dxdy':
                # Multiply by agent velocity
                # Multiply by agent velocity
                # TODO: Only multiply by velocity if we know that the action is a dx,dy
                    # Multiply by agent velocity
                # TODO: Only multiply by velocity if we know that the action is a dx,dy
                    if ind <= num_rovers:
                        input_action_arr=action_arr*config["ccea"]["network"]["rover_max_velocity"]
                    else:
                        input_action_arr=action_arr*config["ccea"]["network"]["uav_max_velocity"]
                elif (config['env']['agents']['rovers']+config['env']['agents']['uavs'])[ind]['action']['type'] == 'pick_uav':
                    # The action arr is the same size as however many uavs there are
                    # We need to get the argmax of this array to tell us which uav to follow
                    # print('action_arr:', action_arr)
                    uav_ind = np.argmax(action_arr)
                    # print('uav_ind:', uav_ind)
                    if uav_ind == num_uavs:
                        # Rover chose the null uav, which means stay still. Don't follow anyone
                        input_action_arr = np.array([0.0, 0.0])
                    else:
                        # Rover chose a uav, so compute a dx,dy where agent makes the biggest step possible towards its chosen uav
                        uav_ind_in_env = int(num_rovers+uav_ind)
                        # print("uav_ind_in_env: ", uav_ind_in_env)
                        uav_position = np.array([env.rovers()[uav_ind_in_env].position().x, env.rovers()[uav_ind_in_env].position().y])
                        agent_position = np.array([env.rovers()[ind].position().x, env.rovers()[ind].position().y])
                        input_action_arr = uav_position - agent_position
                        # Impose velocity boundaries on this delta step
                        if ind < num_rovers: # This is a rover
                            # Impose rover velocity boundaries
                            input_action_arr = bound_velocity_arr(
                                velocity_arr=input_action_arr,
                                max_velocity=config['ccea']['network']['rover_max_velocity']
                            )
                        else: # This is a uav
                            # Impose uav velocity boundaries
                            input_action_arr = bound_velocity_arr(
                                velocity_arr=input_action_arr,
                                max_velocity=config['ccea']['network']['uav_max_velocity']
                            )

                # Save this info for debugging purposes
                # observation_arrs.append(observation_arr)
                actions_arrs.append(input_action_arr)
            for action_arr in actions_arrs:
                action = rovers.tensor(action_arr)
                actions.append(action)
            observations = env.step_without_rewards(actions)

            # Save joint trajectory information
            for i, (action_arr, observation, agent) in enumerate(zip(actions_arrs, observations, env.rovers())):
                if i < num_rovers:
                    type_ = 'rover'
                    agent_id = i
                else:
                    type_ = 'uav'
                    agent_id = i - num_rovers
                joint_trajectory[type_+'_'+str(agent_id)+'_x'].append(agent.position().x)
                joint_trajectory[type_+'_'+str(agent_id)+'_y'].append(agent.position().y)
                joint_trajectory[type_+'_'+str(agent_id)+'_dx'].append(action_arr[0])
                joint_trajectory[type_+'_'+str(agent_id)+'_dy'].append(action_arr[1])
                for j in range(len(observation)):
                    joint_trajectory[type_+'_'+str(agent_id)+'_obs_'+str(j)].append(observation(j))
            for i, poi in enumerate(env.pois()):
                if i < num_rover_pois:
                    type_ = 'rover'
                    poi_index = i
                else:
                    type_ = 'hidden'
                    poi_index = num_rover_pois+i
                joint_trajectory[type_+'_poi_'+str(poi_index)+'_x'].append(poi.position().x)
                joint_trajectory[type_+'_poi_'+str(poi_index)+'_y'].append(poi.position().y)

        # Pad actions
        for i in range(len(env.rovers())):
            if i < num_rovers:
                type_ = 'rover'
                agent_id = i
            else:
                type_ = 'uav'
                agent_id = i - num_rovers
            joint_trajectory[type_+'_'+str(agent_id)+'_dx'].append(None)
            joint_trajectory[type_+'_'+str(agent_id)+'_dy'].append(None)

        if compute_team_fitness:
            # Create an agent pack to pass to reward function
            agent_pack = rovers.AgentPack(
                agent_index = 0,
                agents = env.rovers(),
                entities = env.pois()
            )
            team_fitness = rovers.rewards.Global().compute(agent_pack)
            rewards = env.rewards()
            fitnesses = tuple([r for r in rewards]+[team_fitness])
        else:
            # Each index corresponds to an agent's rewards
            # We only evaulate the team fitness based on the last step
            # so we only keep the last set of rewards generated by the team
            fitnesses = tuple([r for r in rewards])

        return EvalOut(
            fitnesses=fitnesses,
            joint_trajectory=joint_trajectory
        )

    def mutateIndividual(self, individual):
        return tools.mutGaussian(individual, mu=self.config["ccea"]["mutation"]["mean"], sigma=self.config["ccea"]["mutation"]["std_deviation"], indpb=self.config["ccea"]["mutation"]["independent_probability"])

    def mutate(self, population):
        # Only mutate individuals we have set aside for mutation
        for num_individual in range(self.num_mutants):
            mutant_id = num_individual + self.total_elites
            for subpop in population:
                if subpop[0] is not None:
                    self.mutateIndividual(subpop[mutant_id])
                    del subpop[mutant_id].fitness.values

    def selectSubPopulation(self, subpopulation):
        # Assume the persistent elites are best of all time. Leave them alone (for now)
        # new subpop is the offspring of this subpop
        # offspring = subpopulation[:self.n_preserve_elites]
        # small subpop is the subset of subpop that is not persistent
        # small_subpop = subpopulation[self.n_preserve_elites:]

        # Sort elites based on shaped fitness
        sorted_by_individual = sorted(subpopulation, key=lambda ind: ind.fitness.values[0], reverse=True)

        if self.selection_mechanism == 'mixed_n_elites_binary_tournament':
            # Set up lambda function for team sorting
            lambda_func = lambda ind: ind.team_fitness

            # Get the elites based on team fitness
            sorted_by_team = sorted(subpopulation, key=lambda_func, reverse=True)

            # Get persistent team elites
            offspring = sorted_by_team[:self.n_preserve_team_elites]
            # Get persistent individual elites
            offspring += sorted_by_individual[:self.n_preserve_individual_elites]
            # Get current team elites
            offspring += sorted_by_team[self.n_preserve_team_elites:self.n_preserve_team_elites+self.n_team_elites]
            # Get current individual_elites
            offspring += sorted_by_individual[self.n_preserve_individual_elites:self.n_preserve_individual_elites+self.n_individual_elites]

            # Now pick the rest based on a binary tournament.
            # Selection here depends on individual (not team) fitness
            offspring += tools.selTournament(subpopulation, len(subpopulation) - self.total_elites, 2)

        elif self.selection_mechanism == 'epsilon_greedy':
            offspring = []
            for _ in range(self.num_elites):
                if random.uniform(0,1) < self.epsilon:
                    # Select a random agent for survival
                    offspring.append(random.choice(subpopulation))
                else:
                    # Select highest fitness agent for survival
                    offspring.append(sorted_by_individual[0])
            # range(self.num_mutants) ensures we don't produce an extra mutant
            # if the subpopulation is an odd number size, accidentally
            # ballooning the size of the subpopulations
            marked_for_mutation = []
            for individual, _ in zip(offspring, range(self.num_mutants)):
                marked_for_mutation.append(individual)
            offspring += marked_for_mutation

        return [ deepcopy(individual) for individual in offspring ]

    def selectAndMutate(self, populations, team_summaries):
        # Get the correct number of populations first
        offspring = [[] for _ in range(self.num_rovers+self.num_uavs)]

        # Let's get the highest performing teams and individuals
        champion_team_summaries = self.getChampionTeams(team_summaries, self.n_preserve_team_elites+self.n_team_elites)
        for population in populations:
            champion_individuals = tools.selBest(population, self.n_preserve_individual_elites+self.n_individual_elites)

        # Now let's populate offspring with the preserved elites
        preserved_champion_team_summaries = champion_team_summaries[:self.n_preserve_team_elites]
        for team_summary in preserved_champion_team_summaries:
            for i, individual in enumerate(team_summary.individuals):
                offspring[i].append(individual)

        for i, population in enumerate(populations):
            offspring[i] += champion_individuals[:self.n_preserve_individual_elites]

        # Populate with non-preserved elites
        non_preserved_champion_teams = champion_team_summaries[self.n_preserve_team_elites:]
        for team_summary in non_preserved_champion_teams:
            for i, individual in enumerate(team_summary.individuals):
                offspring[i].append(individual)

        for i, population in enumerate(populations):
            offspring[i] += champion_individuals[self.n_preserve_individual_elites:]

        # Populate with mutants from a binary tournament
        for i, population in enumerate(populations):
            mutant_individuals = tools.selTournament(population, len(population)-self.total_elites, 2)
            for individual in mutant_individuals:
                mutant=deepcopy(individual)
                self.mutateIndividual(mutant)
                del mutant.fitness.values
                offspring[i].append(mutant)

        return offspring, preserved_champion_team_summaries

    def select(self, population):
        # Offspring is a list of subpopulation
        offspring = []
        # For each subpopulation in the population
        for subpop in population:
            if subpop[0] is not None:
                # Perform a selection on that subpopulation and add it to the offspring population
                offspring.append(self.selectSubPopulation(subpop))
            else:
                offspring.append(subpop)
        return offspring

    def collectFitnesses(self, eval_team_ins, eval_outs):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness

        for eval_team_in, eval_out in zip(eval_team_ins, eval_outs):
            team_fit = eval_out.fitnesses[-1]
            for individual, shaped_fit in zip(eval_team_in.individuals, eval_out.fitnesses):
                individual.shaped_fitnesses.append(shaped_fit)
                individual.team_fitnesses.append(team_fit)

    def aggregateFitnesses(self, populations):
        for population in populations:
            for individual in population:
                # Reshape the lists of fitnesses to make aggregation easier
                reshape_shaped_fits = [
                    individual.shaped_fitnesses[i * self.num_rollouts_per_team : (i+1) * self.num_rollouts_per_team]
                    for i in range(self.num_teams_per_evaluation)
                ]
                reshape_team_fits = [
                    individual.team_fitnesses[i * self.num_rollouts_per_team : (i+1) * self.num_rollouts_per_team]
                    for i in range(self.num_teams_per_evaluation)
                ]
                # Now aggregate into one value each
                # First aggregate across rollouts
                # Then aggreagte across teams
                agg_shaped_fits_across_rollouts = []
                for shaped_fits_in_rollouts in reshape_shaped_fits:
                    agg_shaped_fits_across_rollouts.append(np.average(shaped_fits_in_rollouts))
                agg_shaped_fit_across_teams = np.average(agg_shaped_fits_across_rollouts)
                # Now for teams
                agg_team_fits_across_rollouts = []
                for team_fits_in_rollouts in reshape_team_fits:
                    agg_team_fits_across_rollouts.append(np.average(team_fits_in_rollouts))
                agg_team_fit_across_teams = np.average(agg_team_fits_across_rollouts)
                # Save the aggregated team and shaped fitness
                individual.fitness.values = (agg_shaped_fit_across_teams,)
                individual.team_fit = agg_team_fit_across_teams

    def buildTeamSummaries(self, team_eval_ins, eval_outs, num_rollouts_per_team):
        team_summaries = []
        for i in range(int(len(eval_outs)/num_rollouts_per_team)):
            team_summaries.append(
                TeamSummary(
                    individuals=team_eval_ins[i*self.num_rollouts_per_team].individuals,
                    seeds=[team_eval_in.seed for team_eval_in in team_eval_ins[i*self.num_rollouts_per_team:(i+1)*self.num_rollouts_per_team]],
                    eval_outs=eval_outs[i * self.num_rollouts_per_team : (i+1) * self.num_rollouts_per_team]
                )
            )
        return team_summaries

    def getChampionTeams(self, team_summaries, num_teams):
        # Aggregate fitnesses across for each team
        aggregated_fitnesses = [
            np.average([eval_out.fitnesses[-1] for eval_out in team_summary.eval_outs])
                for team_summary in team_summaries]
        # Pass this to our helper
        return self.getChampionTeamsHelper(
            team_summaries=team_summaries,
            num_teams=num_teams,
            aggregated_fitnesses=aggregated_fitnesses
        )

    def getChampionTeamsHelper(self, team_summaries, num_teams, aggregated_fitnesses):
        # Base case. No more teams
        if num_teams <= 0:
            return []
        # Choose the highest fitness
        i = np.argmax(aggregated_fitnesses)
        # Get that team as our champion
        champion_team_summary = team_summaries[i]
        return [champion_team_summary] + self.getChampionTeamsHelper(
            team_summaries=team_summaries[:i]+team_summaries[i+1:],
            num_teams=num_teams-1,
            aggregated_fitnesses=aggregated_fitnesses[:i]+aggregated_fitnesses[i+1:]
        )

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createFitnessCsv(self, trial_dir, filename, num_teams, num_rollouts):
        fitness_dir = trial_dir / filename
        # Start the header with generation and overall fits
        header = ['generation', 'overall_fit_team']
        for i in range(self.num_rovers):
            header.append('overall_fit_rover_'+str(i))
        for i in range(self.num_uavs):
            header.append('overall_fit_uav'+str(i))
        # Now iterate through num_teams, and rollouts within a team
        # Each team gets fits aggregated across rollouts, and fits for individual rollouts
        for i in range(num_teams):
            # Aggregated fitnesses
            header.append('agg_fit_team_'+str(i))
            for j in range(self.num_rovers):
                header.append('agg_fit_team_'+str(i)+'_rover_'+str(j))
            for j in range(self.num_uavs):
                header.append('agg_fit_team_'+str(i)+'_uav_'+str(j))
            # Now per rollout
            for r in range(num_rollouts):
                header.append('team_'+str(i)+'_rollout_'+str(r))
                for j in range(self.num_rovers):
                    header.append('team_'+str(i)+'_rollout_'+str(r)+'_rover_'+str(j))
                for j in range(self.num_uavs):
                    header.append('team_'+str(i)+'_rollout_'+str(r)+'_uav_'+str(j))
        # Create the csv
        header_line = ','.join(header)+'\n'
        with open(fitness_dir, 'w') as file:
            file.write(header_line)

    def writeFitnessCsv(self, trial_dir, team_summaries, filename, type):
        # type = 'training', 'team_champion', 'individual_champions'
        fitness_dir = trial_dir / filename
        # We are building a list of fitnesses to append as a line to the csv
        element_list = [self.gen] # (starting with the gen)

        # Get the aggregated team fitnesses
        aggregated_team_fits_across_rollouts = [
            np.average([eval_out.fitnesses[-1] for eval_out in team_summary.eval_outs])
            for team_summary in team_summaries
        ]
        # Get the single fitness value associated with this generation
        aggregated_team_fit_across_teams = np.average(aggregated_team_fits_across_rollouts)

        # Get the aggregated shaped fitnesses
        # [[team_0_shaped_fit_agent_0, team_0_shaped_fit_agent_1 ...], [team_1_shaped_fit_agent_0 ...], [...]]
        aggregated_shaped_fits_across_rollouts = [
            np.average([eval_out.fitnesses[:-1] for eval_out in team_summary.eval_outs], axis=0)
            for team_summary in team_summaries
        ]
        # Get the set of shaped fitnesses associated with this generation
        # [shaped_fit_agent_0, shaped_fit_agent_1, ...]
        aggregated_shaped_fits_across_teams = np.average(aggregated_shaped_fits_across_rollouts, axis=0)

        # Start filling in elements. Starting with the highest-level aggregated values
        element_list.append(aggregated_team_fit_across_teams)
        element_list+=list(aggregated_shaped_fits_across_teams)

        # Next fill in team by team
        for team_summary in team_summaries:
            # Start with the fitnesses aggregated across rollouts for a single team
            element_list.append(np.average([eval_out.fitnesses[-1] for eval_out in team_summary.eval_outs]))
            element_list+=list(np.average([eval_out.fitnesses[:-1] for eval_out in team_summary.eval_outs], axis=0))
            # Then put in the fitnesses from rollouts
            for eval_out in team_summary.eval_outs:
                element_list.append(eval_out.fitnesses[-1])
                element_list+=eval_out.fitnesses[:-1]

        # Make it all into a big comma-separated string, and save it
        csv_line = ','.join([str(i) for i in element_list])+'\n'
        with open(fitness_dir, 'a') as file:
            file.write(csv_line)

    def writeTrajs(self, trial_dir, team_summaries, subfolder, gen):
        top_dir = trial_dir / subfolder
        for i, team_summary in enumerate(team_summaries):
            # Make a directory for this team
            team_dir = top_dir/('team_'+str(i))
            if not os.path.exists(team_dir):
                os.makedirs(team_dir)
            # Make a directory for this gen
            gen_dir = team_dir/('gen_'+str(gen))
            if not os.path.exists(gen_dir):
                os.makedirs(gen_dir)
            for j, eval_out in enumerate(team_summary.eval_outs):
                # Save the joint trajectory
                pd.DataFrame(eval_out.joint_trajectory).to_csv(gen_dir/('rollout'+str(j)+'.csv'), index=False)

    def saveCheckpoint(self, trial_dir, pop):
        checkpoint_dir = trial_dir/('checkpoint_'+str(self.gen)+'.pkl')
        with open(checkpoint_dir, 'wb') as f:
            pickle.dump(pop, f)
        if self.delete_previous_checkpoint:
            checkpoint_dirs = [dir for dir in os.listdir(trial_dir) if "checkpoint_" in dir]
            if len(checkpoint_dirs) > 1:
                lower_gen = min( [int(dir.split("_")[-1].split('.')[0]) for dir in checkpoint_dirs] )
                prev_checkpoint_dir = trial_dir/('checkpoint_'+str(lower_gen)+'.pkl')
                os.remove(prev_checkpoint_dir)

    def getCheckpointDirs(self, trial_dir):
        return [trial_dir/dir for dir in os.listdir(trial_dir) if "checkpoint_" in dir]

    def loadCheckpoint(self, checkpoint_dirs):
        checkpoint_dirs.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        with open(checkpoint_dirs[-1], 'rb') as f:
            pop = pickle.load(f)
        gen = int(str(checkpoint_dirs[-1]).split('_')[-1].split('.')[0])
        return pop, gen

    def evaluatePopulations(self, populations, trial_dir, gen, preserved_team_summaries=[]):
        if gen == 0:
            skip_preserved = False
        else:
            skip_preserved = True

        team_eval_ins = []
        rollout_seeds = [self.get_seed() for _ in range(self.num_rollouts_per_team)]
        for _ in range(self.num_teams_per_evaluation):
            teams = self.formTeams(populations, skip_preserved=skip_preserved)
            for team in teams:
                for _, seed in zip(range(self.num_rollouts_per_team), rollout_seeds):
                    team_eval_ins.append(TeamEvalIn(team, seed))

        # Evaluate the teams
        eval_outs = self.evaluateTeams(team_eval_ins)

        # Collect fitnesses for individuals from team evaluations
        self.collectFitnesses(team_eval_ins, eval_outs)

        # Now aggregate the fitnesses for each individual
        self.aggregateFitnesses(populations)

        # Build up team summaries for recording data
        team_summaries = preserved_team_summaries + self.buildTeamSummaries(team_eval_ins, eval_outs, self.num_rollouts_per_team)

        # Get me the teams with highest performance in the rollouts
        champion_team_summaries = self.getChampionTeams(team_summaries, self.num_team_champions)

        # Resimulate if needed. Otherwise record evaluation info as-is
        if self.resim_test_evaluation:
            champion_team_eval_ins = []
            test_rollout_seeds = [self.get_seed() for _ in range(self.test_num_rollouts)]
            for champion_team_summary in champion_team_summaries:
                for _, seed in zip(range(self.test_num_rollouts), test_rollout_seeds):
                    champion_team_eval_ins.append(TeamEvalIn(champion_team_summary.individuals, seed))
            # Resim champions
            champion_eval_outs = self.evaluateTeams(champion_team_eval_ins)
            # Get the summaries
            resim_champion_team_summaries = self.buildTeamSummaries(champion_team_eval_ins, champion_eval_outs, self.test_num_rollouts)
            # Write fitnesses out to csv
            self.writeFitnessCsv(
                trial_dir=trial_dir,
                team_summaries=resim_champion_team_summaries,
                filename='resim_champion_team_fitness.csv',
                type='champion_teams'
            )
            if self.save_champion_trajectories and self.gen % self.num_gens_between_save_champions == 0:
                # Save trajectories
                self.writeTrajs(trial_dir, resim_champion_team_summaries, subfolder='champion_team')

        else:
            self.writeFitnessCsv(
                trial_dir=trial_dir,
                team_summaries=champion_team_summaries,
                filename='champion_team_fitness.csv',
                type='champion_teams'
            )
            if self.save_champion_trajectories and self.gen % self.num_gens_between_save_champions == 0:
                # Save trajectories
                self.writeTrajs(trial_dir, champion_team_summaries, subfolder='champion_teams', gen=self.gen)

        return team_summaries

    def runTrial(self, num_trial, load_checkpoint):
        # Get trial directory
        trial_dir = self.trials_dir / ("trial_"+str(num_trial))

        # Create directory for saving data
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Check if we're loading from a checkpoint
        if load_checkpoint and len(checkpoint_dirs := self.getCheckpointDirs(trial_dir)) > 0:
            pop, self.gen = self.loadCheckpoint(checkpoint_dirs)
        else:
            # Check if we are starting with a random seed
            if self.random_seed_val is not None:
                # Reset the seed so seed is consistent across trials
                self.reset_seed()
                # unless we specified we want to increment based on the trial number
                if self.increment_seed_every_trial:
                    self.random_seed_val += num_trial
                random.seed(self.get_seed())
                np.random.seed(self.get_seed())

            # Init gen counter
            self.gen = 0

            # Create csv file for saving evaluation fitnesses
            # self.createEvalFitnessCSV(trial_dir)
            # if self.config['data']['save_elite_fitness']['switch']:
            #     self.createEvalFitnessCSV(trial_dir, filename='elite_fitness.csv')
            self.createFitnessCsv(
                trial_dir,
                'champion_team_fitness.csv',
                num_teams=self.num_team_champions,
                num_rollouts=self.test_num_rollouts
            )

            # Initialize the populations
            populations = self.populations()

            # Evaluate populations
            team_summaries = self.evaluatePopulations(populations, trial_dir, self.gen)

        # Don't run anything if we loaded in the checkpoint and it turns out we are already done
        if self.gen >= self.config["ccea"]["num_generations"]:
            return None

        for _ in tqdm(range(self.config["ccea"]["num_generations"]-self.gen)):
            # Update gen counter
            self.gen = self.gen+1

            # Set the seed if one was specified at the start of the generation
            if self.random_seed_val is not None:
                # Reset the seed so seed is consistent across trials
                self.reset_seed()
                # unless we specified we want to increment based on the trial number
                if self.increment_seed_every_trial:
                    self.increment_seed(num_trial)
                # Also increment by generation
                self.increment_seed(self.gen)
                random.seed(self.get_seed())
                np.random.seed(self.get_seed())

            # Perform selection and mutation
            offspring, preserved_champion_team_summaries = self.selectAndMutate(populations, team_summaries)

            # Now populate the population with individuals from the offspring
            self.setPopulation(populations, offspring)

            # Evaluate the new population
            team_summaries[:] = self.evaluatePopulations(populations, trial_dir, self.gen, preserved_champion_team_summaries)

            # # Save checkpoint for generation if now is the time
            # if self.gen == self.config["ccea"]["num_generations"] or \
            #     (self.save_checkpoint and self.gen % self.num_gens_between_checkpoint == 0):
            #     self.saveCheckpoint(trial_dir, populations)

    def run(self, num_trial, load_checkpoint):
        if num_trial is None:
            # Run all trials if no number is specified
            for num_trial in range(self.config["experiment"]["num_trials"]):
                self.runTrial(num_trial, load_checkpoint)
        else:
            # Run only the trial specified
            self.runTrial(num_trial, load_checkpoint)

        if self.use_multiprocessing:
            self.pool.close()

def runCCEA(config_dir, num_trial=None, load_checkpoint=False):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir)
    return ccea.run(num_trial, load_checkpoint)
