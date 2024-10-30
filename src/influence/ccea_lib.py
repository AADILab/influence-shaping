from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm

import multiprocessing
import random
import pickle

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

class JointTrajectory():
    def __init__(self, joint_state_trajectory, joint_observation_trajectory, joint_action_trajectory):
        self.states = joint_state_trajectory
        self.observations = joint_observation_trajectory
        self.actions = joint_action_trajectory

class EvalInfo():
    def __init__(self, fitnesses, joint_trajectory):
        self.fitnesses = fitnesses
        self.joint_trajectory = joint_trajectory

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

        if self.num_rovers > 0:
            self.num_rover_sectors = int(360/self.config["env"]["agents"]["rovers"][0]["resolution"])
            self.rover_nn_template = self.generateTemplateNN(self.num_rover_sectors)
            self.rover_nn_size = self.rover_nn_template.num_weights
        if self.num_uavs > 0:
            self.num_uav_sectors = int(360/self.config["env"]["agents"]["uavs"][0]["resolution"])
            self.uav_nn_template = self.generateTemplateNN(self.num_uav_sectors)
            self.uav_nn_size = self.uav_nn_template.num_weights
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
        return tools.initRepeat(list, self.generateRoverSubpopulation, n=self.num_rovers) + \
            tools.initRepeat(list, self.generateUAVSubpopulation, n=self.num_uavs)

    def formEvaluationTeam(self, population):
        eval_team = []
        for subpop in population:
            # Use max with a key function to get the individual with the highest fitness[0] value
            best_ind = max(subpop, key=lambda ind: ind.fitness.values[0])
            eval_team.append(best_ind)
        return eval_team

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team
        eval_team = self.formEvaluationTeam(population)
        # Evaluate that team however many times we are evaluating teams
        eval_teams = [eval_team for _ in range(self.num_evaluations_per_team)]
        return self.evaluateTeams(eval_teams)

    def formTeams(self, population, inds=None):
        # Start a list of teams
        teams = []

        if inds is None:
            team_inds = range(self.subpopulation_size)
        else:
            team_inds = inds

        # For each individual in a subpopulation
        for i in team_inds:
            # Make a team
            team = []
            # For each subpopulation in the population
            for subpop in population:
                # Put the i'th indiviudal on the team
                team.append(subpop[i])
            # Need to save that team for however many evaluations
            # we're doing per team
            for _ in range(self.num_evaluations_per_team):
                # Save that team
                teams.append(team)

        return teams

    def evaluateTeams(self, teams):
        if self.use_multiprocessing:
            jobs = self.map(self.evaluateTeam, teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.map(self.evaluateTeam, teams))
        return eval_infos

    def evaluateTeam(self, team, compute_team_fitness=True):
        # Set up networks
        rover_nns = [deepcopy(self.rover_nn_template) for _ in range(self.num_rovers)]
        uav_nns = [deepcopy(self.uav_nn_template) for _ in range(self.num_uavs)]

        # Load in the weights
        for rover_nn, individual in zip(rover_nns, team[:self.num_rovers]):
            rover_nn.setWeights(individual)
        for uav_nn, individual in zip(uav_nns, team[self.num_rovers:]):
            uav_nn.setWeights(individual)
        agent_nns = rover_nns + uav_nns

        # Set up the enviornment
        env = createEnv(self.config)

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

        for _ in range(self.num_steps):
            # Compute the actions of all rovers
            observation_arrs = []
            actions_arrs = []
            actions = []
            for ind, (observation, agent_nn) in enumerate(zip(observations, agent_nns)):
                # observation_arr = []
                # for i in range(len(observation)):
                #     observation_arr.append(observation(i))
                # observation_arr = np.array(observation_arr, dtype=np.float64)
                slist = str(observation.transpose()).split(" ")
                flist = list(filter(None, slist))
                nlist = [float(s) for s in flist]
                observation_arr = np.array(nlist, dtype=np.float64)
                action_arr = agent_nn.forward(observation_arr)
                # Multiply by agent velocity
                if ind <= self.num_rovers:
                    action_arr*=self.config["ccea"]["network"]["rover_max_velocity"]
                else:
                    action_arr*=self.config["ccea"]["network"]["uav_max_velocity"]
                # Save this info for debugging purposes
                observation_arrs.append(observation_arr)
                actions_arrs.append(action_arr)
            for action_arr in actions_arrs:
                action = rovers.tensor(action_arr)
                actions.append(action)
            observations, rewards = env.step(actions)

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
        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [ deepcopy(individual) for individual in offspring ]

    def select(self, population):
        # Offspring is a list of subpopulation
        offspring = []
        # For each subpopulation in the population
        for subpop in population:
            # Perform a selection on that subpopulation and add it to the offspring population
            offspring.append(self.selectSubPopulation(subpop))
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
                for individual, fit in zip(team, fitnesses):
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
                for individual, fit in zip(team, fitnesses):
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
                fit_str += ','.join([team_fit]+agent_fits)
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
                    for j in range(self.num_rover_sectors*3):
                        header += "rover_"+str(i)+"_obs_"+str(j)+","
                for i in range(self.num_uavs):
                    for j in range(self.num_uav_sectors*3):
                        header += "uav_"+str(i)+"_obs_"+str(j)+","
                # Actions
                for i in range(self.num_rovers):
                    header += "rover_"+str(i)+"_dx,rover_"+str(i)+"_dy,"
                for i in range(self.num_uavs):
                    header += "uav_"+str(i)+"_dx,uav_"+str(i)+"_dy,"
                header+="\n"
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
