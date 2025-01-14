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

class EvalInfo():
    def __init__(self, fitnesses, joint_trajectory):
        self.fitnesses = fitnesses
        self.joint_trajectory = joint_trajectory

class TeamInfo():
    def __init__(self, policies, seed):
        self.policies = policies
        self.seed = seed

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

        self.n_elites = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["n_elites"]
        self.include_elites_in_tournament = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["include_elites_in_tournament"]
        self.num_mutants = self.subpopulation_size - self.n_elites
        self.num_evaluations_per_team = self.config["ccea"]["evaluation"]["multi_evaluation"]["num_evaluations"]
        self.aggregation_method = self.config["ccea"]["evaluation"]["multi_evaluation"]["aggregation_method"]

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
        self.save_trajectories = self.config["data"]["save_trajectories"]["switch"]
        self.num_gens_between_save_traj = self.config["data"]["save_trajectories"]["num_gens_between_save"]
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
        return random.uniform(self.config["ccea"]["weight_initialization"]["lower_bound"], self.config["ccea"]["weight_initialization"]["upper_bound"])

    def generateIndividual(self, individual_size):
        return tools.initRepeat(creator.Individual, self.generateWeight, n=individual_size)

    def generateRoverIndividual(self):
        return self.generateIndividual(individual_size=self.rover_nn_size)

    def generateUAVIndividual(self):
        return self.generateIndividual(individual_size=self.uav_nn_size)

    def generateRoverSubpopulation(self):
        return tools.initRepeat(list, self.generateRoverIndividual, n=self.config["ccea"]["population"]["subpopulation_size"])

    def generateUAVSubpopulation(self):
        return tools.initRepeat(list, self.generateUAVIndividual, n=self.config["ccea"]["population"]["subpopulation_size"])

    def population(self):
        pop = []
        # Generating subpopulation for each agent
        for agent_id in range(self.num_rovers+self.num_uavs):
            if type(self.template_policies[agent_id]) is NeuralNetwork:
                # Filling subpopulation for each agent
                subpop=[]
                for _ in range(self.config["ccea"]["population"]["subpopulation_size"]):
                    subpop.append(self.generateIndividual(individual_size=self.nn_sizes[agent_id]))
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
        return TeamInfo(policies, self.get_seed())

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team _ times
        eval_teams = [self.formEvaluationTeam(population) for _ in range(self.num_evaluations_per_team)]
        # Evaluate the teams
        return self.evaluateTeams(eval_teams)

    def formTeams(self, population, inds=None) -> List[TeamInfo]:
        # Start a list of teams
        teams = []

        if inds is None:
            team_inds = range(self.subpopulation_size)
        else:
            team_inds = inds

        # For each individual in a subpopulation
        for i in team_inds:
            # Make a team
            policies = []
            # For each subpopulation in the population
            for subpop in population:
                # Put the i'th indiviudal on the team
                policies.append(subpop[i])
            # Need to save that team for however many evaluations
            # we're doing per team
            for _ in range(self.num_evaluations_per_team):
                # Save that team
                teams.append(TeamInfo(policies, self.get_seed()))

        return teams
    
    def buildMap(self, teams):
        return self.map(self.evaluateTeam, teams)
        # if self.use_multiprocessing:
        #     return self.map(
        #         self.evaluateTeam,
        #         zip(
        #             teams,
        #             [self.template_policies for _ in teams],
        #             [self.config for _ in teams],
        #             [self.num_rovers for _ in teams],
        #             [self.num_uavs for _ in teams],
        #             [self.num_steps for _ in teams]
        #         )
        #     )
        # else:
        #     return self.map(
        #         self.evaluateTeam, 
        #         teams,
        #         [self.template_policies for _ in teams],
        #         [self.config for _ in teams],
        #         [self.num_rovers for _ in teams],
        #         [self.num_uavs for _ in teams],
        #         [self.num_steps for _ in teams]
        #     )

    def evaluateTeams(self, teams: List[TeamInfo]):
        if self.use_multiprocessing:
            jobs = self.buildMap(teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.buildMap(teams))
        return eval_infos
    
    def evaluateTeam(self, team: TeamInfo, compute_team_fitness=True):
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
        team: TeamInfo, 
        template_policies: List[Union[NeuralNetwork|FollowPolicy]],
        config: dict,
        num_rovers: int,
        num_uavs: int,
        num_steps: int,
        compute_team_fitness=True
    ):
        # Set up random seed if one was specified
        if not isinstance(team, TeamInfo):
            raise Exception('team should be TeamInfo')
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
        for nn, individual in zip(agent_policies, team.policies):
            if type(nn) is NeuralNetwork:
                nn.setWeights(individual)

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

        if compute_team_fitness:
            # Create an agent pack to pass to reward function
            agent_pack = rovers.AgentPack(
                agent_index = 0,
                agents = env.rovers(),
                entities = env.pois()
            )
            team_fitness = rovers.rewards.Global().compute(agent_pack)
            rewards = env.rewards()
            fitnesses = tuple([(r,) for r in rewards]+[(team_fitness,)])
        else:
            # Each index corresponds to an agent's rewards
            # We only evaulate the team fitness based on the last step
            # so we only keep the last set of rewards generated by the team
            fitnesses = tuple([(r,) for r in rewards])

        return EvalInfo(
            fitnesses=fitnesses,
            joint_trajectory=JointTrajectory(
                joint_state_trajectory,
                joint_observation_trajectory,
                joint_action_trajectory
            )
        )

    def mutateIndividual(self, individual):
        return tools.mutGaussian(individual, mu=self.config["ccea"]["mutation"]["mean"], sigma=self.config["ccea"]["mutation"]["std_deviation"], indpb=self.config["ccea"]["mutation"]["independent_probability"])

    def mutate(self, population):
        # Don't mutate the elites from n-elites
        for num_individual in range(self.num_mutants):
            mutant_id = num_individual + self.n_elites
            for subpop in population:
                if subpop[0] is not None:
                    self.mutateIndividual(subpop[mutant_id])
                    del subpop[mutant_id].fitness.values

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        offspring = tools.selBest(subpopulation, self.n_elites)
        if self.include_elites_in_tournament:
            offspring += tools.selTournament(subpopulation, len(subpopulation)-self.n_elites, tournsize=2)
        else:
            # Get the remaining worse individuals
            remaining_offspring = tools.selWorst(subpopulation, len(subpopulation)-self.n_elites)
            # Add those remaining individuals through a binary tournament
            offspring += tools.selTournament(remaining_offspring, len(remaining_offspring), tournsize=2)
        # Return a deepcopy so that modifying an individual that wasexample/mountain/result/GlobalMultiThreaded selected does not modify every single individual
        # that came from the same selected individual
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
            random.shuffle(subpop)

    def assignFitnesses(self, teams, eval_infos):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        if self.num_evaluations_per_team == 1:
            for team, eval_info in zip(teams, eval_infos):
                fitnesses = eval_info.fitnesses
                for individual, fit in zip(team.policies, fitnesses):
                    individual.fitness.values = fit
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

    def createEvalFitnessCSV(self, trial_dir):
        eval_fitness_dir = trial_dir / "fitness.csv"
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

    def writeEvalFitnessCSV(self, trial_dir, eval_infos):
        eval_fitness_dir = trial_dir / "fitness.csv"
        gen = str(self.gen)
        if len(eval_infos) == 1:
            eval_info = eval_infos[0]
            team_fit = str(eval_info.fitnesses[-1][0])
            agent_fits = [str(fit[0]) for fit in eval_info.fitnesses[:-1]]
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

    def writeEvalTrajs(self, trial_dir, eval_infos):
        # Set up directory
        gen_folder_name = "gen_"+str(self.gen)
        gen_dir = trial_dir / gen_folder_name
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)
        # Iterate through each file we are writing
        for eval_id, eval_info in enumerate(eval_infos):
            eval_filename = "eval_team_"+str(eval_id)+"_joint_traj.csv"
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
                    # for j in range(self.num_rover_sectors*3):
                        header += "rover_"+str(i)+"_obs_"+str(j)+","
                for i in range(self.num_uavs):
                    for j in range(self.num_sensors_uavs[i]):
                    # for j in range(self.num_uav_sectors*3):
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
            self.createEvalFitnessCSV(trial_dir)

            # Initialize the population
            pop = self.population()

            # Create the teams
            teams = self.formTeams(pop)

            # Evaluate the teams
            eval_infos = self.evaluateTeams(teams)

            # Assign fitnesses to individuals
            self.assignFitnesses(teams, eval_infos)

            # Evaluate a team with the best indivdiual from each subpopulation
            eval_infos = self.evaluateEvaluationTeam(pop)

            # Save fitnesses of the evaluation team
            self.writeEvalFitnessCSV(trial_dir, eval_infos)

            # Save trajectories of evaluation team
            if self.save_trajectories:
                self.writeEvalTrajs(trial_dir, eval_infos)

        # Don't run anything if we loaded in the checkpoint and it turns out we are already done
        if self.gen >= self.config["ccea"]["num_generations"]:
            return None

        for _ in tqdm(range(self.config["ccea"]["num_generations"]-self.gen)):
            # Set the seed if one was specified at the start of the generation
            if self.random_seed_val is not None:
                # Reset the seed so seed is consistent across trials
                self.reset_seed()
                # unless we specified we want to increment based on the trial number
                if self.increment_seed_every_trial:
                    self.random_seed_val += num_trial
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

            # Form teams for evaluation
            teams = self.formTeams(offspring)

            # Evaluate each team
            eval_infos = self.evaluateTeams(teams)

            # Now assign fitnesses to each individual
            self.assignFitnesses(teams, eval_infos)

            # Evaluate a team with the best indivdiual from each subpopulation
            eval_infos = self.evaluateEvaluationTeam(offspring)

            # Save fitnesses
            self.writeEvalFitnessCSV(trial_dir, eval_infos)

            # Save trajectories
            if self.save_trajectories and self.gen % self.num_gens_between_save_traj == 0:
                self.writeEvalTrajs(trial_dir, eval_infos)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save checkpoint for generation if now is the time
            if self.gen == self.config["ccea"]["num_generations"] or \
                (self.save_checkpoint and self.gen % self.num_gens_between_checkpoint == 0):
                self.saveCheckpoint(trial_dir, pop)

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
