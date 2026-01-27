from typing import List, Union, Optional, Tuple
from pathlib import Path
from copy import deepcopy
import multiprocessing
import random
import pickle
import random
import os

from tqdm import tqdm
import numpy as np
import yaml
import pandas as pd

from influence.evo_network import NeuralNetwork
from influence.librovers import rover_domain
from influence.custom_env import createEnv
from influence.ccea_utils import  bound_velocity_arr, getRandomWeights, build_rollout_packs
from influence.ccea_utils import FollowPolicy, JointTrajectory, RolloutPackOut, RolloutPackIn, RolloutPack, Individual, TeamPack
from influence.ccea_utils import IndividualPopulation, Team, TeamPopulation, Checkpoint

class CooperativeCoevolutionaryAlgorithm():
    def __init__(self, config_dir):
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = self.config_dir.parent

        with open(str(self.config_dir), 'r') as file:
            self.config = yaml.safe_load(file)

        # Start by setting up variables for different agents
        self.num_rovers = len(self.config["env"]["agents"]["rovers"])
        self.num_uavs = len(self.config["env"]["agents"]["uavs"])
        self.agent_population_size = self.config["ccea"]["populations"]["agent_population_size"]
        self.num_hidden = self.config["ccea"]["network"]["hidden_layers"]

        self.num_rover_pois = len(self.config["env"]["pois"]["rover_pois"])
        self.num_hidden_pois = len(self.config["env"]["pois"]["hidden_pois"])

        self.n_team_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_team_elites']
        self.n_individual_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_individual_elites']
        self.n_individual_elites_based_on_team_fitness = 0
        if 'n_individual_elites_based_on_team_fitness' in self.config['ccea']['selection']['n_elites_binary_tournament']:
            self.n_individual_elites_based_on_team_fitness = self.config['ccea']['selection']['n_elites_binary_tournament']['n_individual_elites_based_on_team_fitness']

        self.num_rollouts_per_team = self.config["ccea"]["evaluation"]["multi_rollout"]["num_rollouts"]
        self.rpt = self.num_rollouts_per_team

        self.num_steps = self.config["ccea"]["num_steps"]
        self.lower_bound = self.config["ccea"]["weight_initialization"]["lower_bound"]
        self.upper_bound = self.config["ccea"]["weight_initialization"]["upper_bound"]
        self.mutation_ind_pb = self.config["ccea"]["mutation"]["independent_probability"]
        self.mutation_mean = self.config["ccea"]["mutation"]["mean"]
        self.mutation_std_dev = self.config["ccea"]["mutation"]["std_deviation"]
        self.num_generations = self.config['ccea']['num_generations']
        self.num_trials = self.config["experiment"]["num_trials"]

        self.template_policies = self.get_template_policies('rovers')+self.get_template_policies('uavs')
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
            elif self.config['env']['agents']['rovers'][i]['sensor']['type'] == 'RoverLidar':
                self.num_sensors_rovers.append(2*int(360/self.config["env"]["agents"]["rovers"][i]["resolution"]))
            elif self.config['env']['agents']['rovers'][i]['sensor']['type'] == 'UavDistanceLidar':
                self.num_sensors_rovers.append(self.num_uavs)
        self.num_sensors_uavs = []
        for i in range(self.num_uavs):
            if self.config["env"]["agents"]["uavs"][i]['sensor']['type'] == 'SmartLidar':
                self.num_sensors_uavs.append(3*int(360/self.config["env"]["agents"]["uavs"][i]["resolution"]))
            elif self.config['env']['agents']['uavs'][i]['sensor']['type'] == 'RoverLidar':
                self.num_sensors_uavs.append(2*int(360/self.config["env"]["agents"]["uavs"][i]["resolution"]))
            elif self.config['env']['agents']['uavs'][i]['sensor']['type'] == 'UavDistanceLidar':
                self.num_sensors_uavs.append(self.num_uavs)

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data saving variables
        self.save_train_trajectories_switch = self.config['data']['save_trajectories']['train']['switch']
        self.num_gens_between_save_train_traj = self.config['data']['save_trajectories']['train']['num_gens_between_save']
        self.save_test_trajectories_switch = self.config['data']['save_trajectories']['test']['switch']
        self.num_gens_between_save_test_traj = self.config['data']['save_trajectories']['test']['num_gens_between_save']

        # Handle train trajectories config
        if 'save_train_trajectories' in self.config["data"]:
            self.save_train_trajectories = self.config["data"]["save_train_trajectories"]["switch"]
            self.num_gens_between_save_train_traj = self.config["data"]["save_train_trajectories"]["num_gens_between_save"]
        else:
            self.save_train_trajectories = False
            self.num_gens_between_save_train_traj = 0

        if 'checkpoints' in self.config['data'] and 'save' in self.config['data']['checkpoints']:
            self.save_checkpoint_switch = self.config["data"]["checkpoints"]["save"]
        else:
            self.save_checkpoint_switch = False
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

        if self.use_multiprocessing:
            self.pool = multiprocessing.Pool(processes=self.num_threads)
            self.map = self.pool.map_async
        else:
            self.map = map

        # Setup uid (unique id) tracking for individuals and teams
        self.individual_uid = 0
        self.team_uid = 0

        self.rovers_cant_move_without_uav = [False for _ in range(self.num_rovers)]
        for rover_id, rover_config in enumerate(self.config['env']['agents']['rovers']):
            if 'needs_uav_to_move' in rover_config:
                self.rovers_cant_move_without_uav[rover_id] = rover_config['needs_uav_to_move']

        # team formation params
        self.team_formation_type = 'random_teams'
        if 'team_formation' in self.config['ccea'] and 'type' in self.config['ccea']['team_formation']:
            self.team_formation_type = self.config['ccea']['team_formation']['type']

        # test team params
        self.test_adhoc_team_switch = False
        if 'test_adhoc_team_switch' in self.config['ccea']['evaluation']:
            self.test_adhoc_team_switch = self.config['ccea']['evaluation']['test_adhoc_team_switch']

    def get_individual_uid(self):
        uid = self.individual_uid
        self.individual_uid+=1
        return uid

    def get_team_uid(self):
        uid = self.team_uid
        self.team_uid+=1
        return uid

    def reset_seed(self):
        """Resets the random seed to what was specified in config"""
        self.random_seed_val = self.config['debug']['random_seed']['set_seed']

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

    def get_template_nn(self, agent_config: dict):
        # Figure out number of inputs to nn from observation size
        if agent_config['sensor']['type'] == 'SmartLidar':
            num_inputs = 3*int(360/agent_config["resolution"])
        elif agent_config['sensor']['type'] == 'UavDistanceLidar':
            num_inputs = self.num_uavs
        elif agent_config['sensor']['type'] == 'RoverLidar':
            num_inputs = 2*int(360/agent_config["resolution"])
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
        return random.uniform(self.lower_bound, self.upper_bound)

    def generateIndividual(self, individual_size, tid):
        return Individual(getRandomWeights(individual_size), tid, self.get_individual_uid())

    def generateRoverIndividual(self):
        return self.generateIndividual(individual_size=self.rover_nn_size)

    def generateUAVIndividual(self):
        return self.generateIndividual(individual_size=self.uav_nn_size)

    def init_agent_populations(self):
        agent_populations = []
        # Generate a population for each agent
        for agent_id in range(self.num_rovers+self.num_uavs):
            if type(self.template_policies[agent_id]) is NeuralNetwork:
                # Filling population for each agent
                pop=[]
                for ind, _ in enumerate(range(self.agent_population_size)):
                    pop.append(self.generateIndividual(individual_size=self.nn_sizes[agent_id], tid=ind))
            else:
                # Population of None for fixed policies
                pop=[]
                for _ in range(self.agent_population_size):
                    pop.append(None)
            agent_populations.append(pop)
        return agent_populations

    def init_team_population(self, agent_populations):
        team_population = []
        for tid in range(self.n_team_elites):
            individuals = []
            for i in range(self.num_rovers+self.num_uavs):
                individuals.append(random.choice(agent_populations[i]))
            team_population.append(Team(individuals,tid,self.get_team_uid()))
        return team_population

    def init_populations(self):
        agent_populations = self.init_agent_populations()
        team_population = self.init_team_population(agent_populations)
        return agent_populations, team_population

    def formEvaluationTeam(self, population):
        policies = []
        for subpop in population:
            if subpop[0] is None:
                best_ind = 0
            else:
                # Use max with a key function to get the individual with the highest shaped fitness
                best_ind = max(subpop, key=lambda ind: ind.shaped_fitness)
            policies.append(best_ind)
        return RolloutPackIn(policies, self.get_seed())

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team _ times
        eval_teams = [self.formEvaluationTeam(population) for _ in range(self.num_rollouts_per_team)]
        # Evaluate the teams
        return self.evaluateTeams(eval_teams)

    def get_rollout_seeds(self):
        seed = self.get_seed()
        if seed is not None:
            seeds = [self.get_seed() for _ in range(self.num_rollouts_per_team)]
        else:
            seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(self.num_rollouts_per_team)]
        return seeds

    def form_teams(self, populations) -> List:
        if self.team_formation_type == 'random_teams':
            return self.form_random_teams(populations)
        elif self.team_formation_type == 'softmax_teams':
            return self.form_softmax_teams(populations)
        elif self.team_formation_type == 'fixed_teams':
            return self.form_shaped_teams(populations)
        else:
            print("WARNING: form_teams did not have a valid formation type specified. Defaulting to random teams.")
            return self.form_random_teams(populations)

    def form_random_teams(self, populations) -> List:
        # Base case: No individuals left
        # For each population
        #   Pick a random policy from the population, excluding preserved elites
        #   Put that policy onto the team
        # Return the team, and the downsized populations with those policies removed

        # Base Case. No individuals left
        if len(populations[0]) <= 0:
            return []

        # Standard Case. Form a team and keep going!
        team = []
        reduced_populations = []
        for pop in populations:
            # Pick anyone!
            index = int(random.choice(np.arange(len(pop))))
            # Add to team
            team.append(pop[index])
            # Remove from team formation
            reduced_populations.append(pop[:index]+pop[index+1:])
        return [team]+self.form_random_teams(reduced_populations)

    def form_shaped_teams(self, populations):
        """Form teams based on putting agents with highest shaped rewards together"""
        # Sort each agent population based on shaped fitness
        sorted_pops = [sorted(pop, key=lambda ind: ind.shaped_fitness if ind.shaped_fitness is not None else 0, reverse=True) for pop in populations]

        # Zip agent populations together into teams,
        # so agents with highest shaped fitness are put together on a team,
        # agents with second highest shaped fitness are on a team, etc
        raw_shaped_teams = []
        for i in range(self.agent_population_size):
            raw_team = [pop[i] for pop in sorted_pops]
            raw_shaped_teams.append(raw_team)
        return raw_shaped_teams

    def form_softmax_teams(self, populations) -> Individual:
        """
        Form teams randomly based on a softmax probability distribution of individuals' shaped fitnesses.
        """

        # Base Case. No individuals left
        if len(populations[0]) <= 0:
            return []

        # Standard Case. Form a team and keep going!
        # Compute probabilities
        pops_probabilities = []
        for pop in populations:
            # Extract fitness values, handling None values
            fitnesses = np.array([ind.shaped_fitness if ind.shaped_fitness is not None else 0.0 for ind in pop])

            # Apply softmax transformation
            # Subtract max for numerical stability
            exp_fitnesses = np.exp(fitnesses - np.max(fitnesses))
            probabilities = exp_fitnesses / np.sum(exp_fitnesses)
            pops_probabilities.append(list(probabilities))

        # Form teams based on probabilities
        team = []
        reduced_populations = []
        for pop, probabilities in zip(populations, pops_probabilities):
            # Pick individual based on softmax distribution
            selected_index = np.random.choice(len(pop), p=probabilities)
            # Add them to the team
            team.append(pop[selected_index])
            # Remove from team formation process
            reduced_populations.append(pop[:selected_index]+pop[selected_index+1:])

        return [team]+self.form_softmax_teams(reduced_populations)

    def build_rollout_pack_ins(self, teams):
        # Get the seeds
        seeds = self.get_rollout_seeds()

        # Need to save that team for however many evaluations
        # we're doing per team
        rollout_pack_ins = []
        for t in teams:
            for s in seeds:
                # Save that team
                rollout_pack_ins.append(RolloutPackIn(t, s))

        return rollout_pack_ins

    def buildMap(self, teams):
        return self.map(self.evaluateTeam, teams)

    def simulate_rollouts(self, rollout_pack_ins: List[RolloutPackIn]):
        if self.use_multiprocessing:
            jobs = self.buildMap(rollout_pack_ins)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.buildMap(rollout_pack_ins))
        return eval_infos

    def evaluateTeam(self, team: RolloutPackIn):
        return self.evaluateTeamStatic(
            team,
            self.template_policies,
            self.config,
            self.num_rovers,
            self.num_uavs,
            self.num_steps
        )

    @staticmethod
    def evaluateTeamStatic(
        rollout_pack_in: RolloutPackIn,
        template_policies: List[Union[NeuralNetwork|FollowPolicy]],
        config: dict,
        num_rovers: int,
        num_uavs: int,
        num_steps: int
    ):
        # Set up random seed if one was specified
        if rollout_pack_in.seed is not None:
            random.seed(rollout_pack_in.seed)
            np.random.seed(rollout_pack_in.seed)

        # Set up networks
        agent_policies = [deepcopy(template_policy) for template_policy in template_policies]

        # Load in the weights
        for nn, individual in zip(agent_policies, rollout_pack_in.individuals):
            if type(nn) is NeuralNetwork:
                nn.setWeights(individual.weights)

        # Setup rover movement constraint
        rovers_cant_move_without_uav = [False for _ in range(num_rovers)]

        for rover_id, rover_config in enumerate(config['env']['agents']['rovers']):
            if 'needs_uav_to_move' in rover_config:
                rovers_cant_move_without_uav[rover_id] = rover_config['needs_uav_to_move']

        # Set up the enviornment
        env = createEnv(config)

        observations, _ = env.reset()

        agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
        poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

        observations_arrs = []
        for observation in observations:
            observation_arr = []
            for i in range(len(observation)):
                observation_arr.append(observation(i))
            observation_arr = np.array(observation_arr, dtype=np.float64)
            observations_arrs.append(observation_arr)

        joint_state_trajectory = [agent_positions+poi_positions]
        joint_observation_trajectory = [observations_arrs]
        joint_action_trajectory = []

        for _ in range(num_steps):
            # Compute the actions of all rovers
            observation_arrs = []
            actions_arrs = []
            actions = []
            for ind, (observation, agent_nn) in enumerate(zip(observations, agent_policies)):
                observation_arr = []
                for i in range(len(observation)):
                    observation_arr.append(observation(i))
                observation_arr = np.array(observation_arr, dtype=np.float64)
                # print("observation_arr:", observation_arr)
                action_arr = agent_nn.forward(observation_arr)
                if (config['env']['agents']['rovers']+config['env']['agents']['uavs'])[ind]['action']['type'] == 'dxdy':
                    if ind <= num_rovers:
                        input_action_arr=action_arr*config["ccea"]["network"]["rover_max_velocity"]
                    else:
                        input_action_arr=action_arr*config["ccea"]["network"]["uav_max_velocity"]
                elif (config['env']['agents']['rovers']+config['env']['agents']['uavs'])[ind]['action']['type'] == 'pick_uav':
                    # The action arr is the same size as however many uavs there are
                    # We need to get the argmax of this array to tell us which uav to follow
                    uav_ind = np.argmax(action_arr)
                    if uav_ind == num_uavs:
                        # Rover chose the null uav, which means stay still. Don't follow anyone
                        input_action_arr = np.array([0.0, 0.0])
                    else:
                        # Rover chose a uav, so compute a dx,dy where agent makes the biggest step possible towards its chosen uav
                        uav_ind_in_env = int(num_rovers+uav_ind)
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

                # Add a hard constraint.
                # If the rover is too far from a uav, it cannot move

                # If this is a rover and it did not sense any uavs, don't move
                if ind < num_rovers \
                    and env.rovers()[ind].m_sensor.num_sensed_uavs() == 0 \
                        and rovers_cant_move_without_uav[ind]:
                    input_action_arr = np.array([0.0,0.0])

                # Save this info for debugging purposes
                observation_arrs.append(observation_arr)
                actions_arrs.append(input_action_arr)
            for action_arr in actions_arrs:
                action = rover_domain.tensor(action_arr)
                actions.append(action)
            # TODO: Might need to start getting the reward at each timestep rather than once at the end
            # And then just summing them together
            # The value tracking for POIs should make this possible
            # And then I guess we do D or D-Ind at each timestep
            # Which should make it possible to do influence tracing
            observations = env.step_without_rewards(actions)

            # Get all the states and all the actions of all agents
            agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
            poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

            joint_observation_trajectory.append(observation_arrs)
            joint_action_trajectory.append(actions_arrs)
            joint_state_trajectory.append(agent_positions+poi_positions)

        # Create an agent pack to pass to reward function
        team_fitness = env.global_reward()
        rewards = env.rewards()
        shaped_fitnesses = [r for r in rewards]

        return RolloutPackOut(
            team_fitness=team_fitness,
            shaped_fitnesses=shaped_fitnesses,
            joint_trajectory=JointTrajectory(
                joint_state_trajectory,
                joint_observation_trajectory,
                joint_action_trajectory
            )
        )

    def sel_tournament(self, population: Union[IndividualPopulation, TeamPopulation], num_offspring: int, tournsize: int):
        offspring = []
        for _ in range(num_offspring):
            competitors = random.sample(population, tournsize)
            if type(population) == IndividualPopulation:
                winner = max(competitors, key=lambda ind: ind.shaped_fitness if ind.shaped_fitness is not None else 0)
            else:
                winner = max(competitors, key=lambda team: team.team_fitness)
            offspring.append(winner)
        return offspring

    def sel_best_individuals(self, population: IndividualPopulation, num_best: int, select_based_on_team_fitness=False):
            if select_based_on_team_fitness:
                return sorted(population, key=lambda ind: ind.team_fitness, reverse=True)[:num_best]
            return sorted(population, key=lambda ind: ind.shaped_fitness, reverse=True)[:num_best]

    def sel_best_teams(self, population: List[TeamPack], num_best: int):
        return sorted(population, key=lambda tp: tp.team.collapsed_team_fitness, reverse=True)[:num_best]

    def mutate_individual(self, individual):
        """Apply Gaussian mutation to an individual"""
        for i in range(len(individual.weights)):
            if random.random() < self.mutation_ind_pb:
                individual.weights[i] += random.gauss(
                    self.mutation_mean,
                    self.mutation_std_dev
                )
            individual.shaped_fitness = None
            individual.team_fitness = None
            individual.rollout_team_fitnesses = None
            individual.rollout_shaped_fitnesses = None
            # Update unique id since this is a new individual
            individual.uid = self.get_individual_uid()
        return individual

    def set_populations(self, agent_populations, agent_offspring_pops, team_populations, team_offspring):
        for agent_pop, agent_offspring in zip(agent_populations, agent_offspring_pops):
            agent_pop[:] = agent_offspring
        team_populations[:] = team_offspring

    def init_fitness_csv(self, trial_dir, filename='fitness.csv'):
        # Build header, start with collapsed fitnesses
        header = ['generation', 'collapsed_team_fitness']
        for r in range(self.num_rovers):
            header.append('collapsed_rover_'+str(r))
        for u in range(self.num_uavs):
            header.append('collapsed_uav_'+str(u))
        # Add rollout fitnesses
        for rollout_id in range(self.num_rollouts_per_team):
            r_prefix = 'rollout_'+str(rollout_id)
            col_name = r_prefix + '_team_fitness'
            header.append(col_name)
            for r in range(self.num_rovers):
                col_name = r_prefix+'_rover_'+str(r)
                header.append(col_name)
            for u in range(self.num_uavs):
                col_name = r_prefix+'_uav_'+str(u)
                header.append(col_name)

        # Write out the csv
        df = pd.DataFrame(columns=header)
        fitness_dir = trial_dir / filename
        df.to_csv(fitness_dir, index=False)

    def update_fitness_csv(self, trial_dir, team_pack: TeamPack, filename: str='fitness.csv')->None:
        # Collapsed fitnesses
        fitness_dict = {
            'generation': self.gen,
            'collapsed_team_fitness': team_pack.team.collapsed_team_fitness
        }
        for r in range(self.num_rovers):
            fitness_dict['collapsed_rover_'+str(r)] = team_pack.team.collapsed_shaped_fitnesses[r]
        for u in range(self.num_uavs):
            idx=self.num_rovers+u
            fitness_dict['collapsed_uav_'+str(u)] = team_pack.team.collapsed_shaped_fitnesses[idx]
        # Rollout fitnesses
        for rollout_id, rollout_pack in enumerate(team_pack.rollout_packs):
            rname = 'rollout_'+str(rollout_id)+'_team_fitness'
            fitness_dict[rname] = rollout_pack.team_fitness
            for r in range(self.num_rovers):
                rname = 'rollout_'+str(rollout_id)+'_rover_'+str(r)
                fitness_dict[rname] = rollout_pack.shaped_fitnesses[r]
            for u in range(self.num_uavs):
                idx=self.num_rovers+u
                rname = 'rollout_'+str(rollout_id)+'_uav_'+str(u)
                fitness_dict[rname] = rollout_pack.shaped_fitnesses[idx]

        df = pd.DataFrame([fitness_dict])
        fitness_dir = trial_dir / filename
        df.to_csv(fitness_dir, mode='a', header=False, index=False)

    def write_trajectories(self, trial_dir, rollout_pack_outs, save_folder=None):
        """Generic function to write trajectories with specified prefix"""
        # Set up directory
        gen_folder_name = "gen_"+str(self.gen)
        dir = trial_dir / gen_folder_name
        if save_folder:
            dir = dir / save_folder
        if not os.path.isdir(dir):
            os.makedirs(dir)
        # Iterate through each file we are writing
        for eval_id, rollout_pack_out in enumerate(rollout_pack_outs):
            eval_filename = "team_"+str(eval_id)+"_joint_traj.csv"
            eval_dir = dir / eval_filename
            with open(eval_dir, 'w') as file:
                # Build up the header (labels at the top of the csv)
                header = ""
                # First the states (agents and POIs)
                for i in range(self.num_rovers):
                    header += "rover_"+str(i)+"_x,rover_"+str(i)+"_y,"
                for i in range(self.num_uavs):
                    header += "uav_"+str(i)+"_x,uav_"+str(i)+"_y,"
                for i in range(self.num_rover_pois):
                    header += "rover_poi_"+str(i)+"_x,rover_poi_"+str(i)+"_y,"
                for i in range(self.num_hidden_pois):
                    header += "hidden_poi_"+str(i)+"_x,hidden_poi_"+str(i)+"_y,"
                # Observations
                for i in range(self.num_rovers):
                    for j in range(self.num_sensors_rovers[i]):
                        header += "rover_"+str(i)+"_obs_"+str(j)+","
                for i in range(self.num_uavs):
                    for j in range(self.num_sensors_uavs[i]):
                        header += "uav_"+str(i)+"_obs_"+str(j)+","
                # Actions
                for i in range(self.num_rovers):
                    header += "rover_"+str(i)+"_dx,rover_"+str(i)+"_dy,"
                for i in range(self.num_uavs):
                    header += "uav_"+str(i)+"_dx,uav_"+str(i)+"_dy,"
                # There will always be a floating comma at the end. Replace it with newline
                header = header[:-1] + '\n'
                # Write out the header at the top of the csv
                file.write(header)
                # Now fill in the csv with the data
                # One line at a time
                joint_traj = rollout_pack_out.joint_trajectory
                # We're going to pad the actions with None because
                # the agents cannot take actions at the last timestep, but
                # there is a final joint state/observations
                action_padding = []
                for action in joint_traj.actions[0]:
                    action_padding.append([None for _ in action])
                joint_traj.actions.append(action_padding)
                for joint_state, joint_observation, joint_action in zip(joint_traj.states, joint_traj.observations, joint_traj.actions):
                    # Aggregate state info
                    state_list = []
                    for state in joint_state:
                        state_list+=[str(state_var) for state_var in state]
                    state_str = ','.join(state_list)
                    # Aggregate observation info
                    observation_list = []
                    for observation in joint_observation:
                        observation_list+=[str(obs_val) for obs_val in observation]
                    observation_str = ','.join(observation_list)
                    # Aggregate action info
                    action_list = []
                    for action in joint_action:
                        action_list+=[str(act_val) for act_val in action]
                    action_str = ','.join(action_list)
                    # Put it all together
                    csv_line = state_str+','+observation_str+','+action_str+'\n'
                    # Write it out
                    file.write(csv_line)

    def writeEvalTrajs(self, trial_dir, eval_infos):
        """Write evaluation trajectories using the generic writeTrajectories function"""
        self.writeTrajectories(trial_dir, eval_infos, prefix="eval_")

    def save_checkpoint(self, trial_dir, agent_pops, team_pop, team_packs):
        checkpoint_dir = trial_dir/('checkpoint_'+str(self.gen)+'.pkl')
        with open(checkpoint_dir, 'wb') as f:
            pickle.dump((agent_pops, team_pop, team_packs), f)
        if self.delete_previous_checkpoint:
            checkpoint_dirs = [dir for dir in os.listdir(trial_dir) if "checkpoint_" in dir]
            if len(checkpoint_dirs) > 1:
                lower_gen = min( [int(dir.split("_")[-1].split('.')[0]) for dir in checkpoint_dirs] )
                prev_checkpoint_dir = trial_dir/('checkpoint_'+str(lower_gen)+'.pkl')
                os.remove(prev_checkpoint_dir)

    def get_checkpoint_dirs(self, trial_dir):
        return [trial_dir/dir for dir in os.listdir(trial_dir) if "checkpoint_" in dir]

    def load_checkpoint(self, checkpoint_dirs):
        checkpoint_dirs.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        with open(checkpoint_dirs[-1], 'rb') as f:
            pop = pickle.load(f)
        gen = int(str(checkpoint_dirs[-1]).split('_')[-1].split('.')[0])
        return pop, gen

    def assign_agent_fitnesses(self, team_packs: List[TeamPack]):
        for team_pack in team_packs:
            for id_, individual in enumerate(team_pack.team.individuals):
                individual.rollout_shaped_fitnesses = [rp.shaped_fitnesses[id_] for rp in team_pack.rollout_packs]
                individual.rollout_team_fitnesses = [rp.team_fitness for rp in team_pack.rollout_packs]
                individual.shaped_fitness = team_pack.team.collapsed_shaped_fitnesses[id_]
                individual.team_fitness = team_pack.team.collapsed_team_fitness

    def build_team_pack(self, rollout_packs: List[RolloutPack], tid: int)->TeamPack:
        return TeamPack(
            team=Team(
                individuals=rollout_packs[0].individuals,
                tid=tid,
                uid=self.get_team_uid()
            ),
            rollout_packs=rollout_packs
        )

    def build_team_packs(self, rollout_packs: List[RolloutPack], num_rollouts_per_team: int) -> List[TeamPack]:
        rpt = num_rollouts_per_team
        rollouts_packs_split_by_team = [
            rollout_packs[i:i + rpt] for i in range(0, len(rollout_packs), rpt)
        ]
        team_packs = [
            self.build_team_pack(rps, tid) for tid, rps in enumerate(rollouts_packs_split_by_team)
        ]
        return team_packs

    def evaluate_populations(self, agent_populations, team_population):
        # Create the teams
        # TODO: Swap out form_softmax_teams for form_teams
        raw_teams = [team.individuals for team in team_population]+self.form_teams(agent_populations)
        rollout_pack_ins = self.build_rollout_pack_ins(raw_teams)

        # Now run the rollouts
        rollout_pack_outs = self.simulate_rollouts(rollout_pack_ins)

        # Save the training trajectories from the rollouts
        if self.save_train_trajectories and self.gen % self.num_gens_between_save_train_traj == 0:
            self.write_trajectories(self.trial_dir, rollout_pack_outs, save_folder='train')

        rollout_packs = build_rollout_packs(rollout_pack_ins, rollout_pack_outs)
        team_packs = self.build_team_packs(rollout_packs, self.num_rollouts_per_team)

        # Assign fitnesses to individuals based on adhoc teams
        adhoc_team_packs = team_packs[self.n_team_elites:]
        self.assign_agent_fitnesses(adhoc_team_packs)

        # Re-simulate our test team (pick from all teams)
        if self.test_adhoc_team_switch:
            test_team_individuals = [
                max(individuals, key=lambda ind: ind.shaped_fitness) for individuals in agent_populations
            ]
        else:
            test_team_individuals = \
                max(team_packs, key=lambda tp: tp.team.collapsed_team_fitness).team.individuals

        seeds = self.get_rollout_seeds()
        test_rollout_pack_ins = [
            RolloutPackIn(individuals=test_team_individuals, seed=seed) for seed in seeds
        ]
        test_rollout_pack_outs = self.simulate_rollouts(test_rollout_pack_ins)
        test_team_pack = self.build_team_pack(
            build_rollout_packs(test_rollout_pack_ins, test_rollout_pack_outs),
            tid=len(team_packs)
        )
        self.update_fitness_csv(self.trial_dir, test_team_pack)

        # Save test trajectories
        if self.save_test_trajectories_switch and self.gen % self.num_gens_between_save_test_traj == 0:
            self.write_trajectories(self.trial_dir, test_rollout_pack_outs, save_folder='test')

        return team_packs

    def generate_offspring(self,
            agent_populations: List[IndividualPopulation],
            team_packs: List[TeamPack]
        ) -> Tuple[List[IndividualPopulation], TeamPopulation]:
        # Put our best teams into team offspring
        team_offspring_pop: List[Team] = deepcopy([tp.team for tp in self.sel_best_teams(team_packs, self.n_team_elites)])

        # Add individuals from teams back into agent populations for selection
        temp_agent_populations = []
        for a_id, agent_pop in enumerate(agent_populations):
            temp_agent_populations.append(agent_pop+[tp.team.individuals[a_id] for tp in team_packs])

        # Get the offspring for agent populations
        agent_offspring_pops: List[IndividualPopulation] = []
        for agent_pop in temp_agent_populations:
            offspring_pop = deepcopy(
                self.sel_best_individuals(
                    agent_pop,
                    self.n_individual_elites
                )
            ) + deepcopy(
                self.sel_best_individuals(
                    agent_pop,
                    self.n_individual_elites_based_on_team_fitness,
                    select_based_on_team_fitness=True
                )
            )
            tourn_select = deepcopy(
                self.sel_tournament(
                    agent_pop,
                    self.agent_population_size-self.n_individual_elites-self.n_individual_elites_based_on_team_fitness,
                    tournsize=2
                )
            )
            for individual in tourn_select:
                self.mutate_individual(individual)
            offspring_pop += tourn_select
            agent_offspring_pops.append(offspring_pop)

        # Assign new tids to all the agents
        for agent_offspring in agent_offspring_pops:
            for tid, individual in enumerate(agent_offspring):
                individual.tid = tid

        # Assign new tids to teams
        for tid, team in enumerate(team_offspring_pop):
            team.tid = tid

        return agent_offspring_pops, team_offspring_pop

    def set_generation_seed(self, num_trial)->None:
        # Set the seed if one was specified in config
        if self.random_seed_val is not None:
            # Reset the seed so seed is consistent across trials
            self.reset_seed()
            # unless we specified we want to increment based on the trial number
            if self.increment_seed_every_trial:
                self.random_seed_val += num_trial
            # Also increment by generation
            self.random_seed_val += self.gen
            random.seed(self.get_seed())
            np.random.seed(self.get_seed())

    def manage_checkpoint_saving(self, agent_pops, team_pop, team_packs)->None:
        if self.gen == self.num_generations or \
            (self.save_checkpoint_switch and self.gen % self.num_gens_between_checkpoint == 0):
            self.save_checkpoint(self.trial_dir, agent_pops, team_pop, team_packs)

    def manage_checkpoint_loading(self, load_checkpoint_switch):
        if load_checkpoint_switch and len(checkpoint_dirs := self.get_checkpoint_dirs(self.trial_dir)) > 0:
            # Try loading the first checkpoint
            # If that throws an error, that means the CCEA was interupted while writing that checkpoint
            # In that case, try the checkpoint before it
            checkpoint_dirs.sort(key=lambda dir: int(str(dir).split('_')[-1].split('.')[0]), reverse=True)
            try:
                idx = 0
                with open(checkpoint_dirs[idx], 'rb') as f:
                    agent_pops, team_pop, team_packs = pickle.load(f)
            except:
                idx = 1
                with open(checkpoint_dirs[idx], 'rb') as f:
                    agent_pops, team_pop, team_packs = pickle.load(f)
            gen = int(str(checkpoint_dirs[idx]).split('_')[-1].split('.')[0])
            return Checkpoint(True, agent_pops, team_pop, team_packs, gen)
        else:
            return Checkpoint(False)

    def init_trial(self, num_trial):
        # Init gen counter
        self.gen = 0

        # Set up seed
        self.set_generation_seed(num_trial)

        # Create csv file for saving testing fitnesses
        self.init_fitness_csv(self.trial_dir)

        # Initialize populations
        agent_pops, team_pop = self.init_populations()

        # Evaluate populations
        team_packs = self.evaluate_populations(agent_pops, team_pop)

        return agent_pops, team_pop, team_packs

    def init_trial_dir(self)->None:
        if not os.path.isdir(self.trial_dir):
            os.makedirs(self.trial_dir)

    def runTrial(self, num_trial, load_checkpoint_switch):
        # Get trial directory
        self.trial_dir = self.trials_dir / ("trial_"+str(num_trial))

        # Create directory for saving data
        self.init_trial_dir()

        # Check if we're loading from a checkpoint
        if (checkpoint := self.manage_checkpoint_loading(load_checkpoint_switch)).loaded:
            agent_pops, team_pop, team_packs, self.gen = \
                checkpoint.agent_pops, checkpoint.team_pop, checkpoint.team_packs, checkpoint.gen
            # Don't run anything if we loaded in the checkpoint and it turns out we are already done
            if self.gen >= self.num_generations:
                return None
        else:
            agent_pops, team_pop, team_packs = self.init_trial(num_trial)

        for _ in tqdm(range(self.num_generations-self.gen)):
            # Update gen counter
            self.gen = self.gen+1

            # Update seed
            self.set_generation_seed(num_trial)

            # Generate offspring (selection and mutation)
            agent_offspring_pops, team_offspring = self.generate_offspring(agent_pops, team_packs)

            # Run evaluation
            team_packs = self.evaluate_populations(agent_offspring_pops, team_offspring)

            # Now set the population with individuals from the offspring
            self.set_populations(agent_pops, agent_offspring_pops, team_pop, team_offspring)

            # Save checkpoint for generation if now is the time
            self.manage_checkpoint_saving(agent_pops, team_pop, team_packs)

    def run(self, num_trial, load_checkpoint):
        if num_trial is None:
            # Run all trials if no number is specified
            for num_trial in range(self.num_trials):
                self.runTrial(num_trial, load_checkpoint)
        else:
            # Run only the trial specified
            self.runTrial(num_trial, load_checkpoint)

        if self.use_multiprocessing:
            self.pool.close()

def runCCEA(config_dir, num_trial=None, load_checkpoint=False):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir)
    return ccea.run(num_trial, load_checkpoint)
