"""
Let's take a step back and factor out the code a bit
Then run some experiments with just the rovers
"""

from deap import base
from deap import creator
from deap import tools
import random
from typing import Union, List
import numpy as np
import multiprocessing
import os
# from scoop import futures

from tqdm import tqdm

from librovers import rovers, thyme
import cppyy

from copy import deepcopy, copy

from custom_env import createEnv

from ccea_toolbox import setupToolbox

if __name__ == "__main__":
    # Set variables
    SUBPOPULATION_SIZE = 50
    INCLUDE_UAVS=False
    REWARD_TYPES=[
        "Difference",
        "Difference",
        "Global",
        "Global"
    ]

    # Let's save data
    save_dir = os.path.expanduser("~")+"/hpc-share/influence/preliminary/2_rovers_2_uavs_G/trial_0"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Create csv files for saving data
    top_line = "generation, team_fitness"
    for i in range(2):
        top_line += ", rover_"+str(i)
    if INCLUDE_UAVS:
        for i in range(2):
            top_line += ", uav_"+str(i)
    for s in range(SUBPOPULATION_SIZE):
        top_line += ", team_fitness_train_"+str(s)
    with open(save_dir+"/fitness.csv", 'w') as file:
        file.write(top_line)
        file.write('\n')

    toolbox = setupToolbox(include_uavs=INCLUDE_UAVS, reward_types=REWARD_TYPES)

    def main():
        # Create population, with subpopulation for each agentpack
        pop = toolbox.population()
        # print(pop)
        # print(len(pop[0]))
        # print(len(pop[1]))
        # print(len(pop[2]))
        # print(len(pop[3]))
        # exit()

        # Define variables for our overall EA
        CXPB = 0.5 # Cross over probability
        MUTPB = 0.2 # Mutation probability
        NGEN = 1000

        # Shuffle each subpopulation
        toolbox.shuffle(pop)

        # Form teams
        teams = toolbox.formTeams(pop)

        # Evaluate each team
        jobs = toolbox.map(toolbox.evaluate, teams)
        team_fitnesses = jobs.get()

        training_fitnesses = []
        # Now we go back through each team and assign fitnesses to individuals on teams
        for team, fitnesses in zip(teams, team_fitnesses):
            # Save the team fitness from training
            training_fitnesses.append(fitnesses[-1][0])
            for individual, fit in zip(team, fitnesses):
                individual.fitness.values = fit

        # Evaluate the champions and save the fitnesses
        fitnesses = toolbox.evaluateBestTeam(pop)
        fit_list = ["0"] + [str(fitnesses[-1][0])] + \
            [str(fit[0]) for fit in fitnesses[:-1]] + \
            [str(fit) for fit in training_fitnesses]
        with open(save_dir+"/fitness.csv", 'a') as file:
            fit_str = ','.join(fit_list)
            file.write(fit_str+'\n')

        # For each generation
        for gen in tqdm(range(NGEN)):
            # Perform a 3-agent tournament selection on each subpopulation
            offspring = toolbox.select(pop)

            # Shuffle the subpopulations
            toolbox.shuffle(pop)

            # Make deepcopies so we don't accidentally overwrite anything
            offspring = list(map(toolbox.clone, offspring))

            # Track which fitnesses are going to be invalid
            invalid_ind = []

            # Crossover
            for num_individual in range(int(SUBPOPULATION_SIZE/2)):
                if random.random() < CXPB:
                    invalid_ind.append(num_individual*2)
                    invalid_ind.append(num_individual*2+1)
                    for subpop in pop:
                        toolbox.crossover(subpop[num_individual*2], subpop[num_individual*2+1])
                        del subpop[num_individual*2].fitness.values
                        del subpop[num_individual*2+1].fitness.values

            # Mutation
            for num_individual in range(SUBPOPULATION_SIZE):
                if random.random() < MUTPB:
                    invalid_ind.append(num_individual)
                    for subpop in pop:
                        toolbox.mutate(subpop[num_individual])
                        del subpop[num_individual].fitness.values

            # Create teams of individuals with invalid fitnesses
            teams = toolbox.formTeams(pop, inds=invalid_ind)

            # Evaluate each team
            jobs = toolbox.map(toolbox.evaluate, teams)
            team_fitnesses = jobs.get()

            # Now we go back through each team and assign fitnesses to individuals on teams
            training_fitnesses = []
            for team, fitnesses in zip(teams, team_fitnesses):
                # Save the team fitness from training
                training_fitnesses.append(fitnesses[-1][0])
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit

            # Evaluate the champions and save the fitnesses
            fitnesses = toolbox.evaluateBestTeam(pop)
            fit_list = [str(gen+1)] + [str(fitnesses[-1][0])] + \
                [str(fit[0]) for fit in fitnesses[:-1]] + \
                [str(fit) for fit in training_fitnesses]
            with open(save_dir+"/fitness.csv", 'a') as file:
                fit_str = ','.join(fit_list)
                file.write(fit_str+'\n')

            # Now populate the population with the individuals from the offspring
            for subpop, subpop_offspring in zip(pop, offspring):
                subpop[:] = subpop_offspring

        return pop

    pop = main()
    for subpop in pop[0:5]:
        for ind in subpop:
            print(ind.fitness.values)
