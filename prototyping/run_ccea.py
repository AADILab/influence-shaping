"""Give this python script a config file and it will run the CCEA using the specified config"""

import argparse
import os
from pathlib import Path
import yaml
import pprint
from tqdm import tqdm

from ccea_toolbox import setupToolbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_ccea.py",
        description="This runs a CCEA according to the given configuration file",
        epilog=""
    )
    parser.add_argument("config_dir")
    args = parser.parse_args()

    config_dir = Path(os.path.expanduser(args.config_dir))
    trials_dir = config_dir.parent

    with open(str(config_dir), 'r') as file:
        config = yaml.safe_load(file)

    # Run for the specified number of trials
    for num_trial in range(config["experiment"]["num_trials"]):
        # Setup the directory for saving data
        trial_dir = trials_dir / ("trial_"+str(num_trial))
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Create csv file for saving data
        fitness_dir = trial_dir / "fitness.csv"
        top_line = "generation, team_fitness"
        for i in range(len(config["env"]["agents"]["rovers"])):
            top_line += ", rover_"+str(i)
        for i in range(len(config["env"]["agents"]["uavs"])):
            top_line += ", uav_"+str(i)
        for s in range(config["ccea"]["population"]["subpopulation_size"]):
            top_line += ", team_fitness_train_"+str(s)
        with open(fitness_dir, 'w') as file:
            file.write(top_line)
            file.write('\n')

        # Setup toolbox for evolution
        toolbox = setupToolbox(config)

        # Create population, with subpopulation for each agentpack
        pop = toolbox.population()

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
        with open( str(fitness_dir) , 'a') as file:
            fit_str = ','.join(fit_list)
            file.write(fit_str+'\n')

        # For each generation
        for gen in tqdm(range(config["ccea"]["num_generations"])):
            # Perform a N-elites binary tournament selection on each subpopulation
            N_ELITES = config["ccea"]["selection"]["n_elites"]
            offspring = toolbox.select(pop, N=N_ELITES)

            # Make deepcopies so we don't accidentally overwrite anything
            offspring = list(map(toolbox.clone, offspring))

            # Track which fitnesses are going to be invalid
            invalid_ind = []

            # Mutation
            SUBPOPULATION_SIZE = config["ccea"]["population"]["subpopulation_size"]
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
            # print("random_teams: ", len(random_teams))

            # Now form teams using the hall of fame for each individual
            hof_teams = toolbox.formHOFTeams(offspring, hall_of_fame_team)
            # print("hof_teams: ", len(hof_teams))

            # Aggregate all the teams
            teams = random_teams + hof_teams
            # print("teams: ", len(teams))
            # exit()

            # Evaluate each team
            jobs = toolbox.map(toolbox.evaluateWithTeamFitness, teams)
            team_fitnesses = jobs.get()

            # Now we go back through each team and assign fitnesses to individuals on teams
            # (This is just based on the fitnesses from the random teams)
            training_fitnesses = []
            # total_individuals = SUBPOPULATION_SIZE*len(offspring)
            num_inds_with_fitness = 0
            for team, fitnesses in zip(teams[:SUBPOPULATION_SIZE], team_fitnesses[:SUBPOPULATION_SIZE]):
                # Save the team fitness from training
                training_fitnesses.append(fitnesses[-1][0])
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit
                    num_inds_with_fitness += 1
            # print("assigned fitness: ", num_inds_with_fitness)

            # exit()

            # Now we are going to add the hall of fame values
            ALPHA = config["ccea"]["evaluation"]["hall_of_fame"]["alpha"]
            individual_index = 0
            for subpop in offspring:
                for individual in subpop:
                    # print("SUBPOPULATION_SIZE: ", SUBPOPULATION_SIZE)
                    # print("individual.fitness.values: ", individual.fitness.values)
                    # print("len(teams): ", len(teams))
                    # print("SUBPOPULATION_SIZE+individual_index: ", SUBPOPULATION_SIZE+individual_index)
                    # print('individual_index: ', individual_index)
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
            with open(str(fitness_dir), 'a') as file:
                fit_str = ','.join(fit_list)
                file.write(fit_str+'\n')

            # Now populate the population with the individuals from the offspring
            for subpop, subpop_offspring in zip(pop, offspring):
                subpop[:] = subpop_offspring
