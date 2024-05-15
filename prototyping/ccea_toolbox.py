from deap import base
from deap import creator
from deap import tools

import multiprocessing
import random

from evo_network import NeuralNetwork
from ccea_utils import evaluate, shuffle, formTeams, formChampionTeam

def setupToolbox(
        num_hidden=[10],
        num_steps=30,
        include_uavs=True,
        subpopulation_size=50,
        use_multiprocessing=True,
        num_threads=20,
        reward_types=["Global", "Global", "Global", "Global"]
    ):
    # Setup variables for convenience
    NUM_ROVERS = 2
    if include_uavs:
        NUM_UAVS = 2
    else:
        NUM_UAVS = 0
    SUBPOPULATION_SIZE = subpopulation_size
    rover_nn = NeuralNetwork(num_inputs=4*3, num_hidden=num_hidden, num_outputs=2)
    uav_nn = NeuralNetwork(num_inputs=12*3, num_hidden=num_hidden, num_outputs=2)
    ROVER_IND_SIZE = rover_nn.num_weights
    UAV_IND_SIZE = uav_nn.num_weights
    NUM_STEPS = num_steps

    # Create the type of fitness we're optimizing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Now set up our evolutionary toolbox
    toolbox = base.Toolbox()
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=num_threads)
        toolbox.register("map", pool.map_async)
    else:
        toolbox.register("map", map)
    toolbox.register("attr_float", random.uniform, -0.5, 0.5)
    # rover or uav individual
    toolbox.register("rover_individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ROVER_IND_SIZE)
    toolbox.register("uav_individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=UAV_IND_SIZE)
    # sub population of rovers or of uavs
    toolbox.register("rover_subpopulation", tools.initRepeat, list, toolbox.rover_individual, n=SUBPOPULATION_SIZE)
    toolbox.register("uav_subpopulation", tools.initRepeat, list, toolbox.uav_individual, n=SUBPOPULATION_SIZE)
    # Custom population function to merge rovers and uavs together in the same overall population
    def population():
        return tools.initRepeat(list, toolbox.rover_subpopulation, n=NUM_ROVERS) + \
            tools.initRepeat(list, toolbox.uav_subpopulation, n=NUM_UAVS)
    # population is a "population" of populations. One for each agent that is co-evolving
    toolbox.register("population", population)

    # Register all of our operators for crossover, mutation, selection, evaluation, team formation
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

    def selectNElitesBinaryTournament(population, N):
        # Get the best N individuals
        offspring = tools.selBest(population, N)

        # Get the remaining worse individuals
        remaining_offspring = tools.selWorst(population, len(population)-N)

        # Add those remaining individuals through a binary tournament
        offspring += tools.selTournament(remaining_offspring, len(remaining_offspring), tournsize=2)

        return offspring

    toolbox.register("selectSubPopulation", selectNElitesBinaryTournament)
    toolbox.register("evaluate", evaluate, num_steps=NUM_STEPS, rover_network=rover_nn, uav_network=uav_nn, include_uavs=include_uavs, reward_types=reward_types)

    def select(population, N):
        # Offspring is a list of subpopulation
        offspring = []
        # For each subpopulation in the population
        for subpop in population:
            # Perform a selection on that subpopulation, and add it to the offspring population
            offspring.append(toolbox.selectSubPopulation(subpop, N))
        return offspring

    toolbox.register("select", select)
    toolbox.register("shuffle", shuffle)
    toolbox.register("formTeams", formTeams)

    def evaluateBestTeam(population):
        """
        Create a champion team that is the best individual from each subpopulation
        Then evaluate that team
        Save the individual rewards (fitness) for each agent
        Then compute a G for the team's performance (team fitness)
        """
        # Create champion team
        champion_team = formChampionTeam(population)
        # Evaluate that team and get the agent fitnesses AND a team fitness
        fitnesses = toolbox.evaluate(champion_team, compute_team_fitness=True)

        return fitnesses

    toolbox.register("evaluateBestTeam", evaluateBestTeam)

    return toolbox
