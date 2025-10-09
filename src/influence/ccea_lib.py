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

from influence.evo_network import NeuralNetwork
from influence.librovers import rovers
from influence.custom_env import createEnv
from influence.ccea_utils import  bound_velocity_arr, getRandomWeights
from influence.ccea_utils import FollowPolicy, JointTrajectory, EvalInfo, TeamInfo, Individual

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

        self.n_preserve_team_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_preserve_team_elites']
        self.n_preserve_individual_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_preserve_individual_elites']
        self.n_team_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_team_elites']
        self.n_individual_elites = self.config['ccea']['selection']['n_elites_binary_tournament']['n_individual_elites']

        self.n_preserve_elites = self.n_preserve_team_elites + self.n_preserve_individual_elites
        self.n_current_elites = self.n_team_elites + self.n_individual_elites

        self.total_elites = self.n_preserve_elites + self.n_current_elites

        self.include_elites_in_tournament = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["include_elites_in_tournament"]
        self.num_mutants = self.subpopulation_size - self.total_elites
        self.num_evaluations_per_team = self.config["ccea"]["evaluation"]["multi_evaluation"]["num_evaluations"]
        self.aggregation_method = self.config["ccea"]["evaluation"]["multi_evaluation"]["aggregation_method"]
        self.sort_teams_by_sum_agent_fitness = False
        if 'sort_teams_by_sum_agent_fitness' in self.config['ccea']['selection']['n_elites_binary_tournament']:
            self.sort_teams_by_sum_agent_fitness = self.config['ccea']['selection']['n_elites_binary_tournament']['sort_teams_by_sum_agent_fitness']

        if 'save_elite_fitness' not in self.config['data']:
            self.config['data']['save_elite_fitness'] = {}
        if 'switch' not in self.config['data']['save_elite_fitness']:
            self.config['data']['save_elite_fitness']['switch'] = False
        self.save_elite_fitness_switch = self.config['data']['save_elite_fitness']['switch']

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
        self.save_trajectories = self.config["data"]["save_trajectories"]["switch"]
        self.num_gens_between_save_traj = self.config["data"]["save_trajectories"]["num_gens_between_save"]

        # Handle train trajectories config
        if 'save_train_trajectories' in self.config["data"]:
            self.save_train_trajectories = self.config["data"]["save_train_trajectories"]["switch"]
            self.num_gens_between_save_train_traj = self.config["data"]["save_train_trajectories"]["num_gens_between_save"]
        else:
            self.save_train_trajectories = False
            self.num_gens_between_save_train_traj = 0

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

        if self.use_multiprocessing:
            self.pool = multiprocessing.Pool(processes=self.num_threads)
            self.map = self.pool.map_async
        else:
            self.map = map

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

    def generateIndividual(self, individual_size, temp_id):
        return Individual(getRandomWeights(individual_size), temp_id)

    def generateRoverIndividual(self):
        return self.generateIndividual(individual_size=self.rover_nn_size)

    def generateUAVIndividual(self):
        return self.generateIndividual(individual_size=self.uav_nn_size)

    def population(self):
        pop = []
        # Generating subpopulation for each agent
        for agent_id in range(self.num_rovers+self.num_uavs):
            if type(self.template_policies[agent_id]) is NeuralNetwork:
                # Filling subpopulation for each agent
                subpop=[]
                for ind, _ in enumerate(range(self.subpopulation_size)):
                    subpop.append(self.generateIndividual(individual_size=self.nn_sizes[agent_id], temp_id=ind))
            else:
                # Subpopulation of None for fixed policies
                subpop=[]
                for _ in range(self.subpopulation_size):
                    subpop.append(None)
            pop.append(subpop)
        return pop

    def formEvaluationTeam(self, population):
        policies = []
        for subpop in population:
            if subpop[0] is None:
                best_ind = 0
            else:
                # Use max with a key function to get the individual with the highest shaped fitness
                best_ind = max(subpop, key=lambda ind: ind.shaped_fitness)
            policies.append(best_ind)
        return TeamInfo(policies, self.get_seed())

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team _ times
        eval_teams = [self.formEvaluationTeam(population) for _ in range(self.num_evaluations_per_team)]
        # Evaluate the teams
        return self.evaluateTeams(eval_teams)

    # def formTeams(self, population, inds=None) -> List[TeamInfo]:
    def formTeams(self, population) -> Tuple[List[TeamInfo], List[int]]:
        # Start a list of teams
        teams = []

        if self.n_preserve_elites > 0 and self.gen != 0:
            # Skip the persistent elites for team formation
            # We want to hold on to these elites and don't want a new fitness assigned to them
            team_inds = [i+self.n_preserve_elites for i in range(self.subpopulation_size-self.n_preserve_elites)]

        else:
            team_inds = list(range(self.subpopulation_size))

        seed = self.get_seed()
        if seed is not None:
            seeds = [self.get_seed() for _ in range(self.num_evaluations_per_team)]
        else:
            seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(self.num_evaluations_per_team)]

        for i in team_inds:
            # Make a team
            policies = []
            # For each subpopulation in the population
            for subpop in population:
                # Put the i'th indiviudal on the team
                policies.append(subpop[i])
            # Need to save that team for however many evaluations
            # we're doing per team
            for s in seeds:
                # Save that team
                teams.append(TeamInfo(policies, s))

        return teams, seeds

    def buildMap(self, teams):
        return self.map(self.evaluateTeam, teams)

    def evaluateTeams(self, teams: List[TeamInfo]):
        if self.use_multiprocessing:
            jobs = self.buildMap(teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.buildMap(teams))
        return eval_infos

    def evaluateTeam(self, team: TeamInfo):
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
        team: TeamInfo,
        template_policies: List[Union[NeuralNetwork|FollowPolicy]],
        config: dict,
        num_rovers: int,
        num_uavs: int,
        num_steps: int
    ):
        # Set up random seed if one was specified
        if not isinstance(team, TeamInfo):
            raise Exception('team should be TeamInfo')
        if team.seed is not None:
            random.seed(team.seed)
            np.random.seed(team.seed)

        # Set up networks
        agent_policies = [deepcopy(template_policy) for template_policy in template_policies]

        # Load in the weights
        for nn, individual in zip(agent_policies, team.policies):
            if type(nn) is NeuralNetwork:
                nn.setWeights(individual.weights)

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

                # Save this info for debugging purposes
                observation_arrs.append(observation_arr)
                actions_arrs.append(input_action_arr)
            for action_arr in actions_arrs:
                action = rovers.tensor(action_arr)
                actions.append(action)
            observations = env.step_without_rewards(actions)

            # Get all the states and all the actions of all agents
            agent_positions = [[agent.position().x, agent.position().y] for agent in env.rovers()]
            poi_positions = [[poi.position().x, poi.position().y] for poi in env.pois()]

            joint_observation_trajectory.append(observation_arrs)
            joint_action_trajectory.append(actions_arrs)
            joint_state_trajectory.append(agent_positions+poi_positions)

        # Create an agent pack to pass to reward function
        agent_pack = rovers.AgentPack(
            agent_index = 0,
            agents = env.rovers(),
            entities = env.pois()
        )
        team_fitness = rovers.rewards.Global().compute(agent_pack)
        rewards = env.rewards()
        fitnesses = [r for r in rewards]+[team_fitness]

        return EvalInfo(
            fitnesses=fitnesses,
            joint_trajectory=JointTrajectory(
                joint_state_trajectory,
                joint_observation_trajectory,
                joint_action_trajectory
            )
        )

    def selTournament(self, population, num_offspring, tournsize):
        offspring = []
        for _ in range(num_offspring):
            competitors = random.sample(population, tournsize)
            winner = max(competitors, key=lambda ind: ind.shaped_fitness if ind.shaped_fitness is not None else 0)
            offspring.append(winner)
        return offspring

    def mutateIndividual(self, individual):
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
        return individual

    def mutate(self, population):
        # Don't mutate the elites from n-elites
        for num_individual in range(self.num_mutants):
            mutant_id = num_individual + self.total_elites
            for subpop in population:
                if subpop[0] is not None:
                    self.mutateIndividual(subpop[mutant_id])

    def selectSubPopulation(self, subpopulation):
        # Assume the persistent elites are best of all time. Leave them alone (for now)
        # new subpop is the offspring of this subpop
        # offspring = subpopulation[:self.n_preserve_elites]
        # small subpop = the subset of subpop that is not persistent
        # small_subpop = subpopulation[self.n_preserve_elites:]

        # Set up lambda function for team sorting
        if self.sort_teams_by_sum_agent_fitness:
            lambda_func = lambda ind: ind.agg_team_fitness
        else:
            lambda_func = lambda ind: ind.team_fitness

        # Get the elites based on team fitness
        sorted_by_team = sorted(subpopulation, key=lambda_func, reverse=True)
        # Get the elites based on individual fitness
        sorted_by_individual = sorted(subpopulation, key=lambda ind: ind.shaped_fitness, reverse=True)

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
        offspring += self.selTournament(subpopulation, len(subpopulation) - self.total_elites, 2)

        return [ deepcopy(individual) for individual in offspring ]

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

    def shuffle(self, population):
        for subpop in population:
            # If we're preserving elites, then shuffle everyone else
            if self.n_preserve_elites:
                non_elites = subpop[self.n_preserve_elites:]
                random.shuffle(non_elites)
                # Replace the entire subpop with elites and shuffled non-elites
                subpop[:] = subpop[:self.n_preserve_elites] + non_elites
            # Otherwise, shuffle everyone, including elites
            else:
                random.shuffle(subpop)

    def assignFitnesses(self, teams, eval_infos):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        if self.num_evaluations_per_team == 1:
            for team, eval_info in zip(teams, eval_infos):
                fitnesses = eval_info.fitnesses
                team.fitness = eval_info.fitnesses[-1]
                # Agg fitness is the aggregation of all shaped fitnesses on the team
                team.agg_fitness = sum(fit for fit in fitnesses)
                for individual, fit in zip(team.policies, fitnesses):
                    if individual is not None:
                        individual.shaped_fitness = fit
                        individual.team_fitness = team.fitness
                        individual.agg_team_fitness = team.agg_fitness
        else:
            team_list = []
            eval_info_list = []
            for team, eval_info in zip(teams, eval_infos):
                team_list.append(team)
                eval_info_list.append(eval_info)

            for team_id, team in enumerate(team_list[::self.num_evaluations_per_team]):
                # Get all the eval infos for this team
                team_eval_infos = eval_info_list[team_id*self.num_evaluations_per_team:(team_id+1)*self.num_evaluations_per_team]
                # Aggregate the fitnesses into a big numpy array
                all_fitnesses = [eval_info.fitnesses for eval_info in team_eval_infos]
                average_fitnesses = [0 for _ in range(len(all_fitnesses[0]))]
                for fitnesses in all_fitnesses:
                    for count, fit in enumerate(fitnesses):
                        average_fitnesses[count] += fit[0]
                for ind in range(len(average_fitnesses)):
                    average_fitnesses[ind] = average_fitnesses[ind] / self.num_evaluations_per_team
                # And now get that back to the individuals
                fitnesses = tuple([(f,) for f in average_fitnesses])
                for individual, fit in zip(team.policies, fitnesses):
                    if individual is not None:
                        individual.fitness.values = fit

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, trial_dir, filename='fitness.csv'):
        eval_fitness_dir = trial_dir / filename
        header = "generation,team_fitness_aggregated"
        for j in range(self.num_rovers):
            header += ",rover_"+str(j)+"_"
        for j in range(self.num_uavs):
            header += ",uav_"+str(j)
        for i in range(self.num_evaluations_per_team):
            header+=",team_fitness_"+str(i)
            for j in range(self.num_rovers):
                header+=",team_"+str(i)+"_rover_"+str(j)
            for j in range(self.num_uavs):
                header+=",team_"+str(i)+"_uav_"+str(j)
        header += "\n"
        with open(eval_fitness_dir, 'w') as file:
            file.write(header)

    def writeEvalFitnessCSV(self, trial_dir, eval_infos, filename='fitness.csv'):
        eval_fitness_dir = trial_dir / filename
        gen = str(self.gen)
        if len(eval_infos) == 1:
            eval_info = eval_infos[0]
            team_fit = str(eval_info.fitnesses[-1])
            agent_fits = [str(fit) for fit in eval_info.fitnesses[:-1]]
            fit_list = [gen, team_fit]+agent_fits
            fit_str = ','.join(fit_list)+'\n'
        else:
            team_eval_infos = []
            for eval_info in eval_infos:
                team_eval_infos.append(eval_info)
            # Aggergate the fitnesses into a big numpy array
            num_ind_per_team = len(team_eval_infos[0].fitnesses)
            all_fit = np.zeros(shape=(self.num_evaluations_per_team, num_ind_per_team))
            for num_eval, eval_info in enumerate(team_eval_infos):
                fitnesses = eval_info.fitnesses
                for num_ind, fit in enumerate(fitnesses):
                    all_fit[num_eval, num_ind] = fit[0]
                all_fit[num_eval, -1] = fitnesses[-1][0]
            # Now compute a sum/average/min/etc dependending on what config specifies
            agg_fit = np.average(all_fit, axis=0)
            # And now record it all, starting with the aggregated one
            agg_team_fit = str(agg_fit[-1])
            agg_agent_fits = [str(fit) for fit in agg_fit[:-1]]
            fit_str = gen+','+','.join([agg_team_fit]+agg_agent_fits)+','
            # And now add all the fitnesses from individual trials
            # Each row should have the fitnesses for an evaluation
            for row in all_fit:
                team_fit = str(row[-1])
                agent_fits = [str(fit) for fit in row[:-1]]
                fit_str += ','.join([team_fit]+agent_fits) + ','
            fit_str+='\n'
        # Now save it all to the csv
        with open(eval_fitness_dir, 'a') as file:
                file.write(fit_str)

    def writeTrajectories(self, trial_dir, eval_infos, prefix="eval_"):
        """Generic function to write trajectories with specified prefix"""
        # Set up directory
        gen_folder_name = "gen_"+str(self.gen)
        gen_dir = trial_dir / gen_folder_name
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)
        # Iterate through each file we are writing
        for eval_id, eval_info in enumerate(eval_infos):
            eval_filename = prefix + "team_"+str(eval_id)+"_joint_traj.csv"
            eval_dir = gen_dir / eval_filename
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
                joint_traj = eval_info.joint_trajectory
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

    def evaluatePopulations(self, pop, sort_teams=True):
        # Create the teams
        teams, seeds = self.formTeams(pop)

        # Evaluate the teams
        eval_infos = self.evaluateTeams(teams)

        # Assign fitnesses to individuals
        self.assignFitnesses(teams, eval_infos)

        # Save training trajectories if enabled
        if self.save_train_trajectories and self.gen % self.num_gens_between_save_train_traj == 0:
            self.writeTrajectories(self.trial_dir, eval_infos, prefix="train_")

        if sort_teams:
            # Organize subpopulations by individual with highest team fitness first
            if self.sort_teams_by_sum_agent_fitness:
                lambda_func = lambda ind: ind.agg_team_fitness
            else:
                lambda_func = lambda ind: ind.team_fitness
            for subpop in pop:
                subpop.sort(key = lambda_func, reverse=True)

        # Evaluate a team with the best indivdiual from each subpopulation
        eval_infos = self.evaluateEvaluationTeam(pop)

        # Save fitnesses of the evaluation team
        self.writeEvalFitnessCSV(self.trial_dir, eval_infos)

        if self.save_elite_fitness_switch:
            eval_infos = [self.evaluateTeam(
                team=TeamInfo(
                    policies=[subpop[0] for subpop in pop],
                    seed=seeds[0]
                )
            )]
            self.writeEvalFitnessCSV(self.trial_dir, eval_infos, filename='elite_fitness.csv')

        # Save trajectories of evaluation team
        if self.save_trajectories and self.gen % self.num_gens_between_save_traj == 0:
            self.writeEvalTrajs(self.trial_dir, eval_infos)

    def runTrial(self, num_trial, load_checkpoint):
        # Get trial directory
        self.trial_dir = self.trials_dir / ("trial_"+str(num_trial))

        # Create directory for saving data
        if not os.path.isdir(self.trial_dir):
            os.makedirs(self.trial_dir)

        # Check if we're loading from a checkpoint
        if load_checkpoint and len(checkpoint_dirs := self.getCheckpointDirs(self.trial_dir)) > 0:
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
            self.createEvalFitnessCSV(self.trial_dir)
            if self.save_elite_fitness_switch:
                self.createEvalFitnessCSV(self.trial_dir, filename='elite_fitness.csv')

            # Initialize the population
            pop = self.population()

            self.evaluatePopulations(pop)

        # Don't run anything if we loaded in the checkpoint and it turns out we are already done
        if self.gen >= self.num_generations:
            return None

        for _ in tqdm(range(self.num_generations-self.gen)):
            # Set the seed if one was specified at the start of the generation
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

            # Update gen counter
            self.gen = self.gen+1

            # Perform selection
            offspring = self.select(pop)

            # Perform mutation
            self.mutate(offspring)

            # Shuffle subpopulations in offspring
            # to make teams random
            self.shuffle(offspring)

            # Run evaluation
            self.evaluatePopulations(offspring)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save checkpoint for generation if now is the time
            if self.gen == self.num_generations or \
                (self.save_checkpoint and self.gen % self.num_gens_between_checkpoint == 0):
                self.saveCheckpoint(self.trial_dir, pop)

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
