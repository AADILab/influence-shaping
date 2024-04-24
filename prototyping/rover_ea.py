"""
This is a minimal (as possible) example of using an EA from deap on the rover domain. One rover navigating to POIs.
I followed the deap tutorial here: https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
"""
from deap import base
from deap import creator
from deap import tools
import random
from typing import Union, List
import numpy as np

from tqdm import tqdm

from librovers import rovers, thyme

# Need to create functionality to
# - create a neural network with num_hidden_units and num_hidden_layers
# - tell me how many parameters that is
# - load in parameters from a list into the weights of the network
# - run a forward pass using those parameters

# We want to learn a neural network, so here is a network class
class NeuralNetwork:
    def __init__(self, num_inputs: int, num_hidden: Union[int, List[int]], num_outputs: int) -> None:
        if type(num_hidden) == int:
            num_hidden = [num_hidden]
        self.num_inputs, self.num_hidden, self.num_outputs = num_inputs, num_hidden, num_outputs
        # Number of nodes in each layer
        self.shape = tuple([self.num_inputs] + self.num_hidden + [self.num_outputs])
        # Number of layers
        self.num_layers = len(self.shape) - 1
        # Initialize weights with zeros
        self.weights = self.initWeights()
        # Store shape and size for later
        self.num_weights = self.calculateNumWeights()
        # self.weights_shape = calculateWeightShape(self.weights)
        # self.total_weights = calculateWeightSize(self.weights)

    def shape(self):
        return (self.num_inputs, self.num_hidden, self.num_outputs)

    def initWeights(self) -> List[np.ndarray]:
        """Creates the numpy arrays for holding weights. Initialized to zeros """
        weights = []
        for num_inputs, num_outputs in zip(self.shape[:-1], self.shape[1:]):
            weights.append(np.zeros(shape=(num_inputs+1, num_outputs)))
        return weights

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Input layer is not activated.
        # We treat it as an activated layer so that we don't activate it.
        # (you wouldn't activate an already activated layer)
        a = X
        # Feed forward through each layer of hidden units and the last layer of output units
        for layer_ind in range(self.num_layers):
            # Add bias term
            b = np.hstack((a, [1]))
            # Feedforward through the weights w. summations
            f = b.dot(self.weights[layer_ind])
            # Activate the summations
            a = self.activation(f)
        return a

    def setWeights(self, list_of_weights: List[float])->None:
        """Take a list of weights and set the
        neural network weights according to these weights"""
        # Check the size
        if len(list_of_weights) != self.num_weights:
            raise Exception("Weights are being set incorrectly in setWeights().\n"\
                            "The number of weights in the list is not the same as\n"\
                            "the number of weights in the network\n"\
                            +str(len(list_of_weights))+"!="+str(self.num_weights))
        list_ind = 0
        for layer in self.weights:
            for row in layer:
                for element_ind in range(row.size):
                    row[element_ind] = list_of_weights[list_ind]
                    list_ind+=1

    def getWeights(self) -> List[float]:
        """Get the weights as a list"""
        weight_list = []
        for layer in self.weights:
            for row in layer:
                for element in row:
                    weight_list.append(element)
        return weight_list

    def activation(self, arr: np.ndarray) -> np.ndarray:
        return np.tanh(arr)

    def calculateNumWeights(self) -> int:
        return sum([w.size for w in self.weights])

# Let's make a network so we can get the size figured out
toy_nn = NeuralNetwork(num_inputs=8, num_hidden=[10], num_outputs=2)
IND_SIZE = toy_nn.num_weights

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -0.5, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Create a rover evaluation (This is where things get interesting)
def evaluate(individual, num_steps, network: NeuralNetwork):
    """Load up a neural network with the weights from the individual
    Then set up a rover environment
    And run the rover environment for the specified steps
    Give us the reward of the rover at the end (very sparse,
    high reward if the rover ends the episode near a POI)

    """

    # Set up network for the rover
    rover_nn = network
    rover_nn.setWeights(individual)

    # set up the rover
    Dense = rovers.Lidar[rovers.Density]
    Discrete = thyme.spaces.Discrete
    Global = rovers.rewards.Global
    rover_obs_radius = 3.0
    agents = [rovers.Rover[Dense, Discrete, Global](rover_obs_radius, Dense(90), Global())]
    # agents = [rovers.Rover[Dense, Discrete](1.0, Dense(90))]

    # Set up the POIs. This sets the value of the POI to 1
    poi_positions = [
        [9.0, 9.0],
        [5.0, 5.0],
        [9.0, 1.0],
        [5.0, 1.0]
    ]
    num_pois = len(poi_positions)
    # Create the POI objects, each with value of 1
    countConstraint = rovers.CountConstraint(1)
    poi_value = 1.0
    poi_obs_radius = 3.0
    pois = [rovers.POI[rovers.CountConstraint](poi_value,poi_obs_radius,countConstraint) for _ in range(num_pois)]
    agent_positions = [
        [1.0, 1.0]
    ]

    Env = rovers.Environment[rovers.CustomInit]
    env = Env(rovers.CustomInit(agent_positions, poi_positions), agents, pois)

    states, rewards = env.reset()

    total_agent_reward = rewards[0]
    for _ in range(num_steps):
        # Compute the actions of all the rovers (in this case just one rover)
        actions = []
        for state in states:
            slist = str(state.transpose()).split(" ")
            flist = list(filter(None, slist))
            nlist = [float(s) for s in flist]
            state_arr = np.array(nlist, dtype=np.float64)
            action_arr = rover_nn.forward(state_arr)
            action = rovers.tensor(action_arr)
            actions.append(action)
        # Step forward the environment with those actions
        states, rewards = env.step(actions)
        # Save the reward to the cumulative reward
        total_agent_reward += rewards[0]

    # if total_agent_reward > 4:
    #     print("Anomaly: ", total_agent_reward, reward_list)
    return total_agent_reward,

# Register all of our operators for crossover, mutation, selection, and evaluation
toolbox.register("crossover", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, num_steps=100, network=toy_nn)

def main():
    # Create our population with n=50 random individuals
    pop = toolbox.population(n=50)

    # Define variables for our overall EA
    CXPB = 0.5 # Cross over probability
    MUTPB = 0.2 # Mutation probability
    NGEN = 1000

    # Evaluate the entire population
    # Apply the evaluate function to each invididual in the population
    # and aggregate the output of that function (individual fitness) into
    # a list of all the fitnesses of the population
    fitnesses = map(toolbox.evaluate, pop)

    # Once we have aggregated those fitnesses, go back and assign
    # each fitness to the corresponding individual in the population
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    # Do the following loop for the specified number of genfied individuals have their fitness invalidated. The individuals are cloned so returned population is independent of the input erations
    for _ in tqdm(range(NGEN)):
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
