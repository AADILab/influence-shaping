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

from ccea_toolbox import setupToolbox

for t in range(20):
    if __name__ == "__main__":
        # Set variables
        SUBPOPULATION_SIZE = 50
        INCLUDE_UAVS=True
        REWARD_TYPES=[
            "Difference",
            "Difference",
            "Global",
            "Global"
        ]
        N_ELITES = 5
        ALPHA = 0.5

        # Let's save data
        save_dir = os.path.expanduser("~")+"/hpc-share/influence/preliminary/coding/trial_"+str(t)
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

        toolbox = setupToolbox(include_uavs=INCLUDE_UAVS, reward_types=REWARD_TYPES, use_multiprocessing=True)

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
            NGEN = 1000

            # Form teams
            teams = toolbox.formTeams(pop)

            # Evaluate each team
            jobs = toolbox.map(toolbox.evaluateWithTeamFitness, teams)
            team_fitnesses = jobs.get()

            # Save the Hall of Fame champion team
            team_pairs = zip(teams, team_fitnesses)
            hall_of_fame_team_pair = max(team_pairs, key=lambda team_pair: team_pair[1][-1][0])
            hall_of_fame_team = hall_of_fame_team_pair[0]

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
                # Perform a N-elites binary tournament selection on each subpopulation
                offspring = toolbox.select(pop, N=N_ELITES)

                # Make deepcopies so we don't accidentally overwrite anything
                offspring = list(map(toolbox.clone, offspring))

                # Track which fitnesses are going to be invalid
                invalid_ind = []

                # Mutation
                for num_individual in range(SUBPOPULATION_SIZE-N_ELITES):
                    # if random.random() < MUTPB:
                    invalid_ind.append(num_individual+N_ELITES)
                    for subpop in offspring:
                        toolbox.mutate(subpop[num_individual+N_ELITES])
                        del subpop[num_individual+N_ELITES].fitness.values

                # Shuffle subpopulations in the offspring
                toolbox.shuffle(offspring)

                # Form random teams of individuals
                random_teams = toolbox.formTeams(offspring)

                # Now form teams using the hall of fame for each individual
                hof_teams = toolbox.formHOFTeams(offspring, hall_of_fame_team)

                # Aggregate all the teams
                teams = random_teams + hof_teams

                # Evaluate each team
                jobs = toolbox.map(toolbox.evaluateWithTeamFitness, teams)
                team_fitnesses = jobs.get()

                # Now we go back through each team and assign fitnesses to individuals on teams
                # (This is just based on the fitnesses from the random teams)
                training_fitnesses = []
                # total_individuals = SUBPOPULATION_SIZE*len(offspring)
                for team, fitnesses in zip(teams[:SUBPOPULATION_SIZE], team_fitnesses[:SUBPOPULATION_SIZE]):
                    # Save the team fitness from training
                    training_fitnesses.append(fitnesses[-1][0])
                    for individual, fit in zip(team, fitnesses):
                        individual.fitness.values = fit

                # Now we are going to add the hall of fame values
                individual_index = 0
                for subpop in offspring:
                    for individual in subpop:
                        # print("total_individuals: ", total_individuals)
                        # print("individual.fitness.values: ", individual.fitness.values)
                        # print("len(teams): ", len(teams))
                        # print("total_individuals+individual_index: ", total_individuals+individual_index)
                        individual.fitness.values = \
                            (individual.fitness.values[0]*ALPHA + (1-ALPHA)*team_fitnesses[SUBPOPULATION_SIZE+individual_index][-1][0],)
                        individual_index+=1

                # And now we check if there is a new hall of fame team
                team_pairs = zip(teams, team_fitnesses)
                new_hall_of_fame_team_pair = max(team_pairs, key=lambda team_pair: team_pair[1][-1][0])
                if new_hall_of_fame_team_pair[1][-1][0] > hall_of_fame_team_pair[1][-1][0]:
                    hall_of_fame_team_pair = new_hall_of_fame_team_pair
                    hall_of_fame_team = new_hall_of_fame_team_pair[0]

                # Save the fitnesses of the HOF team
                fitnesses = hall_of_fame_team_pair[1]
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