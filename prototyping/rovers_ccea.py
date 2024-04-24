from deap import base
from deap import creator
from deap import tools
import random
from typing import Union, List
import numpy as np

from tqdm import tqdm

from librovers import rovers, thyme

from copy import deepcopy, copy

# Neural network class for evaluation
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
SUBPOPULATION_SIZE=50
NUM_AGENTS=2
NUM_STEPS=1

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -0.5, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
# sub population is a "population" of individuals
toolbox.register("subpopulation", tools.initRepeat, list, toolbox.individual, n=SUBPOPULATION_SIZE)
# population is a "population" of populations. One for each agent that is co-evolving
toolbox.register("population", tools.initRepeat, list, toolbox.subpopulation, n=NUM_AGENTS)

# Create a multi-rover evaluation
def evaluate(team, num_steps, network: NeuralNetwork):
    """Load in the rovers and evaluate them

    team: list of individuals (each individual is a list of weights)
    """

    # Create a neural network for each rover
    rover_nns = [deepcopy(network) for _ in range(len(team))]
    # Load the weights for each rover (individual) into that rover's neural network
    for rover_nn, individual in zip(rover_nns, team):
        rover_nn.setWeights(individual)

    # Set up the rovers
    Dense = rovers.Lidar[rovers.Density]
    Discrete = thyme.spaces.Discrete
    Difference = rovers.rewards.Difference
    agents = [rovers.Rover[Dense, Discrete, Difference](1.0, Dense(90), Difference()) for _ in range(len(team))]

    # Set up POI positions
    poi_positions = [
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0]
    ]
    # Save the number of POIS
    num_pois = len(poi_positions)
    # Create the POI objects, each with value of 1
    countConstraint = rovers.CountConstraint(1)
    pois = [rovers.POI[rovers.CountConstraint](1,1,countConstraint) for _ in range(num_pois)]
    # pois = [rovers.POI[rovers.CountConstraint](1) for _ in range(num_pois)]
    # Now set the coupling (count constraint) of each POI to 1
    # for poi in pois:
    #     poi.m_constraint.count_constraint = 1

    # Set up the agent positions
    agent_positions = [
        [2.0, 2.0],
        [4.0, 4.0],
        # [2.0, 2.0],
        # [2.0, 2.0]
    ]

    # Set up an environment with those rovers
    Env = rovers.Environment[rovers.CustomInit]
    env = Env(rovers.CustomInit(agent_positions, poi_positions), agents, pois)

    # env.set_rovers(agents)
    # env.set_pois(pois)
    
    states, rewards = env.reset()

    for _ in range(num_steps):
        # Compute the actions of all the rovers (in this case just one rover)
        actions = []
        for state, rover_nn in zip(states, rover_nns):
            slist = str(state.transpose()).split(" ")
            flist = list(filter(None, slist))
            nlist = [float(s) for s in flist]
            state_arr = np.array(nlist, dtype=np.float64)
            action_arr = rover_nn.forward(state_arr)
            action = rovers.tensor(action_arr)
            actions.append(action)

        # Step forward the environment with those actions
        states, rewards = env.step(actions)

    # Each index corresponds to an agent's rewards
    return tuple([(reward,) for reward in rewards])

# Register all of our operators for mutation, selection, evaluation, team formation
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
toolbox.register("selectSubPopulation", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate, num_steps=NUM_STEPS, network=toy_nn)

def select(population):
    # Offspring is a list of subpopulation
    offspring = []
    # For each subpopulation in the population
    for subpop in population:
        # Perform a selection on that subpopulation, and add it to the offspring population
        offspring.append(toolbox.selectSubPopulation(subpop, len(subpop)))
    return offspring

toolbox.register("select", select)

def randomTeams(population):
    # Shuffle each subpopulation (this is an in-place operation)
    for subpop in population:
        random.shuffle(subpop)
    
    # Start a list of teams
    teams = []
    # For each individual in a subpopulation
    for i in range(len(subpop)):
        # Make a team
        team = []
        # For each subpopulation in the population
        for subpop in population:
            # Put the i'th indiviudal on the team
            team.append(subpop[i])
        # Save that team
        teams.append(team)
    
    return teams

toolbox.register("randomTeams", randomTeams)

def main():
    # Create population, with subpopulation for each agentpack
    pop = toolbox.population()

    # Define variables for our overall EA
    NGEN = 1000

    # Create random teams for evaluation
    teams = toolbox.randomTeams(pop)

    # Evaluate each team
    team_fitnesses = map(toolbox.evaluate, teams)

    # Now we go back through each team and assign fitnesses to individuals on teams
    for team, fitnesses in zip(teams, team_fitnesses):
        for individual, fit in zip(team, fitnesses):
            individual.fitness.values = fit

    # For each generation
    for _ in tqdm(range(NGEN)):
        # Perform a binary tournament selection on each subpopulation
        offspring = toolbox.select(pop)

        # Mutate all of the offspring 
        for subpop in offspring:
            for individual in subpop:
                # then mutate this "mutant"
                toolbox.mutate(individual)
                # And delete its fitness values because this mutant
                # has different parameters now
                del individual.fitness.values

        # Create random teams for evaluation
        teams = toolbox.randomTeams(pop)

        # Evaluate each team
        team_fitnesses = map(toolbox.evaluate, teams)

        # Now we go back through each team and assign fitnesses to individuals on teams
        for team, fitnesses in zip(teams, team_fitnesses):
            for individual, fit in zip(team, fitnesses):
                individual.fitness.values = fit

        # Now populate the population with the individuals from the offspring
        for subpop, subpop_offspring in zip(pop, offspring):
            subpop[:] = subpop_offspring

    return pop

if __name__ == "__main__":
    pop = main()
    for subpop in pop[0:5]:
        for ind in subpop:
            print(ind.fitness.values)
