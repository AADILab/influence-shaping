from deap import base
from deap import creator
from deap import tools
import random

# Need to create a fitness class
# Positive weight means "maximize this objective"
# https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness.weights
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Now we create an individual that we are going to evolve.
# Really, this is neural network, but to fully leverage
# the power of deap, we are going to represent the parameters
# a list until we evaluate the individual. Then we will load the parameters into a network
creator.create("Individual", list, fitness=creator.FitnessMax)

# The toolbox is what we use for helpers in deap
toolbox = base.Toolbox()

# Every time the toolbox calls "attr_float", it will actually be calling random.random
# (This is an alias)
toolbox.register("attr_float", random.random)

# This is setting up a function alias
# Whenever the toolbox calls "individual", it will actually be calling "tools.initRpeat"
# with the following arguments: container=creator.Individual, func=toolbox.attr_float, number of repititions = IND_SIZE
# So whenever we create an individual, we are actually using other tools from deap to build that individual
IND_SIZE = 30
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)

# Now we want an easy way to build out a populuation of individuals
# When we call "population", run tools.initRepeat to create a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Create a simple evaluation function for this example
def evaluate(individual):
    """Return the sum of the floating point numbers in the individual"""
    return sum(individual),

# Register all of our operators for crossover, mutation, selection, and evaluation
toolbox.register("crossover", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    # Create our population with n=50 random individuals
    pop = toolbox.population(n=50)
    
    # Define variables for our overall EA
    CXPB = 0.5 # Cross over probability
    MUTPB = 0.2 # Mutation probability
    NGEN = 100

    # Evaluate the entire population
    # Apply the evaluate function to each invididual in the population
    # and aggregate the output of that function (individual fitness) into
    # a list of all the fitnesses of the population
    fitnesses = map(toolbox.evaluate, pop)

    # Once we have aggregated those fitnesses, go back and assign
    # each fitness to the corresponding individual in the population
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    # Do the following loop for the specified number of generations
    for g in range(NGEN):
        # Select the next generation individuals using the tournament
        # selection operator we specified earlier
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals 
        # (Make 1:1 copies so we don't accidentally overwrite any
        # data while we are still using that data)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        # The indexing for offspring is set up so that every two offspring
        # Are going to iteratively be set to child1, child2
        # So each odd indexed offspring will be child1, and the corresponding
        # even indexed offspring will be child2
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # If we randomly draw floating point number between 0 and 1
            # that is lower than the crossover probability:
            if random.random() < CXPB:
                # Perform crossover on these offspring
                toolbox.crossover(child1, child2)
                # Delete their fitnesses because they 
                # now have new parameters
                del child1.fitness.values
                del child2.fitness.values

        # Now go through all of the offspring (after crossover)
        for mutant in offspring:
            # If we randomly draw number smaller than the mutation probability
            if random.random() < MUTPB:
                # then mutate this "mutant"
                toolbox.mutate(mutant)
                # And delete its fitness values because this mutant
                # has different parameters now
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # (the ones whose fitnesses we deleted during crossover and mutation)
        # Get a list of all of the individuals in the offspring with invalid fitnesses
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # Now use the evaluate function to evaluate each individual with an invalid
        # fitness and save all those fitnesses
        fitnesses = map(toolbox.evaluate, invalid_ind)
        # Now let's assign each fitness to the individual that fitness is for
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # The population is entirely replaced by the offspring
        # (pop may have some additional class methods and attributes that we
        # don't want to get rid of, which is why I think we use [:] to make
        # sure we are replacing all the elements in the list, but not the entire
        # object)
        pop[:] = offspring

    return pop

if __name__ == "__main__":
    pop = main()
    for ind in pop:
        print(ind.fitness.values)
