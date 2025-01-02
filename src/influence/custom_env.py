from typing import List
from influence.librovers import rovers, thyme
import numpy as np
import cppyy
import random

def listToVec(list_: List[int]):
    return cppyy.gbl.std.vector[cppyy.gbl.int](list_)

def calculateDistance(position_0, position_1):
    return np.linalg.norm([position_0.x-position_1.x, position_0.y-position_1.y])

def calculateAngle(position_0, position_1):
    pos0 = np.array([position_0.x, position_0.y])
    pos1 = np.array([position_1.x, position_1.y])
    # Create a vector from position 0 to 1
    vec = pos1 - pos0
    # Take the arctan2 of the y, x of that vector
    angle = np.arctan2(vec[1], vec[0]) * 180./np.pi
    if angle < 0:
        angle += 360.
    return angle

# Agents should be able to distinguish between rovers and uavs...
# Unfortunately this means writing my own Lidar class that can differentiate between agent types
class SmartLidar(rovers.Lidar[rovers.Density]):
    def __init__(self, resolution, composition_policy, agent_types, poi_types):
        super().__init__(resolution, composition_policy)
        self.agent_types = agent_types
        self.poi_types = poi_types
        self.m_resolution = resolution
        self.m_composition = composition_policy

    def scan(self, agent_pack):
        # print("SmartLidar.scan()")
        num_sectors = int(360. / self.m_resolution)
        poi_values = [[] for _ in range(num_sectors)]
        rover_values = [[] for _ in range(num_sectors)]
        uav_values = [[] for _ in range(num_sectors)]
        agent = agent_pack.agents[agent_pack.agent_index]
        my_type = self.agent_types[agent_pack.agent_index]

        # Observe POIs
        # print("Observe POIs")
        for poi_ind, sensed_poi in enumerate(agent_pack.entities):
            # print("Sensing POI")
            # Get angle and distance to POI
            angle = calculateAngle(agent.position(), sensed_poi.position())
            distance = calculateDistance(agent.position(), sensed_poi.position())
            # print("angle: ", angle)
            # print("distance: ", distance)
            # Skip this POI if it is out of range
            if distance > agent.obs_radius():
                # print("continue, out of range")
                continue
            # Skip this POI if I am not capable of observing this POI
            if my_type == "rover" and self.poi_types[poi_ind] == "hidden":
                continue
            # Bin the POI according to where it was sensed
            if angle < 360.0:
                sector = int( angle / self.m_resolution )
            else:
                sector = 0
            # print("sector: ", sector, type(sector))
            poi_values[sector].append(sensed_poi.value() / max([0.001, distance**2]))
        # print("Finished POI sensing")

        # print("Observe Agents")
        # Observe agents
        # print('agent_pack.agents.size() | ', agent_pack.agents.size())
        # print('len(self.agent_types) | ', len(self.agent_types))
        for i in range(agent_pack.agents.size()):
            # print("Sensing agent")
            # print('i | ', i)
            # Do not observe yourself
            if i == agent_pack.agent_index:
                # print("Nope, that one is me")
                continue
            # Get angle and distance to sensed agent
            sensed_agent = agent_pack.agents[i]
            # print('sensed_agent | ', sensed_agent)
            angle = calculateAngle(agent.position(), sensed_agent.position())
            # print("angle: ", angle)
            distance = calculateDistance(agent.position(), sensed_agent.position())
            # print("distance: ", distance)
            # Skip the agent if the sensed agent is out of range
            if distance > agent.obs_radius():
                # print("continue, out of range")
                continue
            # Get the bin for this agent
            if angle < 360.0:
                sector = int( angle / self.m_resolution )
            else:
                sector = 0
            # print("Got bin for this agent")
            # Bin the agent according to type
            if self.agent_types[i] == "rover":
                rover_values[sector].append(1.0 / max([0.001, distance**2]))
            elif self.agent_types[i] == "uav":
                uav_values[sector].append(1.0 / max([0.001, distance**2]))
            # print("Finished binning agent")
        # print("Finished agent sensing")

        # print("rover_values: ", rover_values)
        # print("uav_values: ", uav_values)
        # print("poi_values: ", poi_values)

        # Encode the state
        # print("Encoding state")
        state = np.array([-1.0 for _ in range(num_sectors*3)])
        # print("state: ", state)
        for i in range(num_sectors):
            # print("Building sector ", i)
            num_rovers = len(rover_values[i])
            num_uavs = len(uav_values[i])
            num_pois = len(poi_values[i])

            if num_rovers > 0:
                # print("num_rovers > 0")
                # print("rover_values["+str(i)+"]: ", rover_values[i], type(rover_values[i]), type(rover_values[i][0]))
                # print("num_rovers: ", type(num_rovers))
                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()
                for r in rover_values[i]:
                    cpp_vector.push_back(r)
                state[i] = self.m_composition.compose(cpp_vector, 0.0, num_rovers)
            if num_uavs > 0:
                # print("num_uavs > 0")
                # print("uav_values["+str(i)+"]: ", uav_values[i], type(uav_values[i]), type(uav_values[i][0]))
                # print("num_uavs: ", type(num_uavs))
                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()
                for u in uav_values[i]:
                    cpp_vector.push_back(u)
                state[num_sectors + i] = self.m_composition.compose(cpp_vector, 0.0, num_uavs)
            if num_pois > 0:
                # print("num_pois > 0")
                # print("poi_values["+str(i)+"]: ", poi_values[i], type(poi_values[i]), type(poi_values[i][0]))
                # print("num_pois: ", type(num_pois))
                # Convert poi_values[i] to a std::vector<double> to satisfy cppyy
                # Not sure why this is necessary sometimes and other times not necessary
                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()
                for p in poi_values[i]:
                    cpp_vector.push_back(p)
                state[num_sectors*2 + i] = self.m_composition.compose(cpp_vector, 0.0, num_pois)

        return rovers.tensor(state)

# First we're going to create a simple rover
def createRover(obs_radius, reward_type, resolution, agent_types, poi_types):
    Discrete = thyme.spaces.Discrete
    Reward = rovers.rewards.Global
    type_ = 'rover'
    # TODO: Add another parameter here so that you can specify different parameters for indirect difference rewards?
    # Maybe even a set of parameters? Just throw in a hashmap directly in there (of all D-Indirec related parameters)? 
    # Rather than putting each D-Indirect parameter in individually?
    rover = rovers.Rover[SmartLidar, Discrete, Reward](reward_type, type_, obs_radius, SmartLidar(resolution=resolution, composition_policy=rovers.Density(), agent_types=agent_types, poi_types=poi_types), Reward())
    return rover

# Now create a UAV
def createUAV(obs_radius, reward_type, resolution, agent_types, poi_types):
    Discrete = thyme.spaces.Discrete
    Reward = rovers.rewards.Global
    type_ = 'uav'
    uav = rovers.Rover[SmartLidar, Discrete, Reward](reward_type, type_, obs_radius, SmartLidar(resolution=resolution, composition_policy=rovers.Density(), agent_types=agent_types, poi_types=poi_types), Reward())
    return uav

def createAgent(agent_config, agent_types, poi_types, type_):
    """Create an agent using the agent's config and type"""
    # unpack config
    reward_type = agent_config['reward_type']
    obs_radius = agent_config['observation_radius']
    resolution = agent_config['resolution']

    # repackage indirect difference parameters
    IndirectDifferenceParameters = rovers.IndirectDifferenceParameters
    AutomaticParameters = rovers.AutomaticParameters
    if 'IndirectDifference' in agent_config:
        indirect_difference_config = agent_config['IndirectDifference']
        auto_params_config = indirect_difference_config['automatic']
        indirect_difference_parameters = IndirectDifferenceParameters(
            type_ = indirect_difference_config['type'],
            assignment = indirect_difference_config['assignment'],
            manual = listToVec(indirect_difference_config['manual']) if 'manual' in indirect_difference_config else [],
            automatic_parameters = AutomaticParameters(
                timescale = auto_params_config['timescale'],
                credit = auto_params_config['credit']
            )
        )
        
    else:
        # Use default if none are specified
        indirect_difference_parameters = IndirectDifferenceParameters(
            type_ = 'removal',
            assignment = 'automatic',
            manual = cppyy.gbl.std.vector[cppyy.gbl.int](),
            automatic_parameters = AutomaticParameters(
                timescale = 'trajectory',
                credit = 'AllOrNothing'
            )
        )

    Discrete = thyme.spaces.Discrete
    Reward = rovers.rewards.Global

    return rovers.Rover[SmartLidar, Discrete, Reward](
        indirect_difference_parameters,
        reward_type,
        type_,
        obs_radius,
        SmartLidar(
            resolution=resolution, 
            composition_policy=rovers.Density(), 
            agent_types=agent_types, 
            poi_types=poi_types
        ),
        Reward()
    )

# Now create a POI constraint where this POI can only be observed by rovers with "rover" type
class AbstractRoverConstraint(rovers.IConstraint):
    def __init__(self, coupling, is_rover_list):
        super().__init__()
        self.coupling = coupling
        self.is_rover_list = is_rover_list

    def captured(self, dist, agent, entity):
        """Tell us if the agent observed the poi (entity) from specified distance (dist)"""
        if entity.capture_radius() != -1.0 and dist <= entity.capture_radius():
            return True
        elif dist <= agent.obs_radius() and dist <= entity.obs_radius():
            return True
        else:
            return False
    
    def _step_is_satisfied(self, entity_pack, t):
        count = 0
        dists = []
        constraint_satisfied = False
        for agent in entity_pack.agents:
            if agent.type() == 'rover':
                if agent.path()[t].x == -1 and agent.path()[t].y == -1:
                    # [-1, -1] means this rover was counterfactually removed
                    dist = np.inf
                else:
                    dist = calculateDistance(agent.path()[t], entity_pack.entity.position())
                dists.append(dist)
                if self.captured(dist, agent, entity_pack.entity):
                    count += 1
                    if count >= self.coupling:
                        constraint_satisfied = True
        if constraint_satisfied:
            dists.sort()
            dists = [max(1.0, dist) for dist in dists]
            constraint_value = float(self.coupling)
            for dist in dists[:self.coupling]:
                constraint_value = constraint_value*1.0/dist
            # print("constraint_value: ", constraint_value)
            return constraint_value
        return 0.0

class RoverConstraint(AbstractRoverConstraint):
    """Constraint based on final positions"""
    def is_satisfied(self, entity_pack):
        # print("RoverConstraint.is_satisfied()")
        # print(f"RoverConstraint.is_satisfied() | entity_pack.agents.size() | {entity_pack.agents.size()}")
        # No agents means constraint is not satisfied
        if entity_pack.agents.size() == 0:
            return 0.0
        else:
            t_final = entity_pack.agents[0].path().size()-1
            # print(f"RoverConstraint.is_satisfied() | t_final | {t_final}")
            # print(f"RoverConstraint.is_satisfied() | ")
            return self._step_is_satisfied(entity_pack, t=t_final)

class RoverSequenceConstraint(AbstractRoverConstraint):
    """Constraint based on closest positions in paths"""
    def is_satisfied(self, entity_pack):
        if entity_pack.agents.size() == 0:
            return 0.0
        else:
            steps = []
            for t in range(entity_pack.agents[0].path().size()):
                steps.append(self._step_is_satisfied(entity_pack, t))
            return max(steps)

def createRoverPOI(value, obs_rad, capture_radius, coupling, is_rover_list, constraint):
    if constraint == 'sequential':
        roverConstraint = RoverSequenceConstraint(coupling, is_rover_list)
        poi = rovers.POI[RoverSequenceConstraint](value, obs_rad, capture_radius, roverConstraint)
    elif constraint == 'final':
        roverConstraint = RoverConstraint(coupling, is_rover_list)
        poi = rovers.POI[RoverConstraint](value, obs_rad, capture_radius, roverConstraint)
    return poi

# This is just to help me track which POIs are nominally hidden from rovers
def createHiddenPOI(*args, **kwargs):
    return createRoverPOI(*args, **kwargs)

# Running into errors setting up the environment
# Let's try it with regular POIs
def createPOI(value, obs_rad, coupling, is_rover_list):
    countConstraint = rovers.CountConstraint(coupling)
    poi = rovers.POI[rovers.CountConstraint](value, obs_rad, countConstraint)
    return poi

def resolvePositionSpawnRule(position_dict):
    if position_dict["spawn_rule"] == "fixed":
        return position_dict["fixed"]
    elif position_dict["spawn_rule"] == "random_uniform":
        low_x = position_dict["random_uniform"]["low_x"]
        high_x = position_dict["random_uniform"]["high_x"]
        x = random.uniform(low_x, high_x)
        low_y = position_dict["random_uniform"]["low_y"]
        high_y = position_dict["random_uniform"]["high_y"]
        y = random.uniform(low_y, high_y)
        return [x,y]
    elif position_dict["spawn_rule"] == "random_circle":
        theta = random.uniform(0, 2*np.pi)
        r = position_dict["random_circle"]["radius"]
        center = position_dict["random_circle"]["center"]
        x = r*np.cos(theta)+center[0]
        y = r*np.sin(theta)+center[1]
        return [x,y]

# Let's have a function that builds out the environment
def createEnv(config):
    Env = rovers.Environment[rovers.CustomInit]

    # Aggregate all of the positions of agents
    agent_positions = []
    for rover in config["env"]["agents"]["rovers"]:
        position = resolvePositionSpawnRule(rover["position"])
        agent_positions.append(position)
    for uav in config["env"]["agents"]["uavs"]:
        position = resolvePositionSpawnRule(uav["position"])
        agent_positions.append(position)

    # Aggregate all of the positions of pois
    poi_positions = []
    for rover_poi in config["env"]["pois"]["rover_pois"]:
        position = resolvePositionSpawnRule(rover_poi["position"])
        poi_positions.append(position)
    for hidden_poi in config["env"]["pois"]["hidden_pois"]:
        position = resolvePositionSpawnRule(hidden_poi["position"])
        poi_positions.append(position)

    NUM_ROVERS = len(config["env"]["agents"]["rovers"])
    NUM_UAVS = len(config["env"]["agents"]["uavs"])
    NUM_ROVER_POIS = len(config["env"]["pois"]["rover_pois"])
    NUM_HIDDEN_POIS = len(config["env"]["pois"]["hidden_pois"])

    agent_types = ["rover"]*NUM_ROVERS + ["uav"]*NUM_UAVS
    poi_types = ["rover"]*NUM_ROVER_POIS+["hidden"]*NUM_HIDDEN_POIS

    # rovers_ = [
    #     createRover(
    #         obs_radius=rover["observation_radius"],
    #         reward_type=rover["reward_type"],
    #         resolution=rover["resolution"],
    #         agent_types=agent_types,
    #         poi_types=poi_types
    #     )
    #     for rover in config["env"]["agents"]["rovers"]
    # ]
    rovers_ = [
        createAgent(
            agent_config=rover_config, 
            agent_types=agent_types,
            poi_types=poi_types,
            type_='rover'
        )
        for rover_config in config['env']['agents']['rovers']
    ]
    # uavs = [
    #     createUAV(
    #         obs_radius=uav["observation_radius"],
    #         reward_type=uav["reward_type"],
    #         resolution=uav["resolution"],
    #         agent_types=agent_types,
    #         poi_types=poi_types
    #     )
    #     for uav in config["env"]["agents"]["uavs"]
    # ]
    uavs = [
        createAgent(
            agent_config=uav_config,
            agent_types=agent_types,
            poi_types=poi_types,
            type_ = 'uav'
        )
        for uav_config in config['env']['agents']['uavs']
    ]
    agents = rovers_+uavs

    is_rover_list = [True if str_=="rover" else False for str_ in agent_types]

    # Fill defaults for new features
    for poi_config in config['env']['pois']['rover_pois']+config['env']['pois']['hidden_pois']:
        if 'constraint' not in poi_config:
            poi_config['constraint'] = 'final'
        if 'capture_radius' not in poi_config:
            poi_config['capture_radius'] = -1.0

    rover_pois = [
        createRoverPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            capture_radius=poi["capture_radius"],
            coupling=poi["coupling"],
            is_rover_list=is_rover_list,
            constraint=poi['constraint']
        )
        for poi in config["env"]["pois"]["rover_pois"]
    ]
    hidden_pois = [
        createHiddenPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            capture_radius=poi["capture_radius"],
            coupling=poi["coupling"],
            is_rover_list=is_rover_list,
            constraint=poi['constraint']
        )
        for poi in config["env"]["pois"]["hidden_pois"]
    ]
    pois = rover_pois + hidden_pois

    debug_reward_equals_G = False
    if 'debug' in config and 'reward_equals_G' in config['debug']:
        debug_reward_equals_G = config['debug']['reward_equals_G']

    env = Env(
        rovers.CustomInit(agent_positions, poi_positions),
        agents,
        pois,
        width=cppyy.gbl.ulong(config["env"]["map_size"][0]),
        height=cppyy.gbl.ulong(config["env"]["map_size"][1]),
        debug_reward_equals_G = debug_reward_equals_G
    )
    return env

# Alright let's give this a try
def main():
    config = {
        "env": {
            "agents": {
                "rovers": [
                    {
                        "observation_radius": 3.0,
                        "reward_type": "Global",
                        "resolution": 90,
                        "position": [10.0, 10.0]
                    }
                ],
                "uavs": [
                    {
                        "observation_radius": 100.0,
                        "reward_type": "Global",
                        "resolution": 90,
                        "position": [25.0, 25.0]
                    }
                ]
            },
            "pois": {
                "rover_pois": [
                    {
                        "value": 1.0,
                        "observation_radius": 1.0,
                        "coupling": 1,
                        "position": [40, 40.0],
                    }
                ],
                "hidden_pois": [
                    {
                        "value": 5.0,
                        "observation_radius": 5.0,
                        "coupling": 1,
                        "position": [13.0, 10.0]
                    }
                ]
            },
            "map_size": [50.0, 50.0]
        }
    }

    env = createEnv(config)

    states, rewards = env.reset()

    print("States:")
    for ind, state in enumerate(states):
        print("agent "+str(ind))
        print(state.transpose())

    print("Rewards:")
    for ind, reward in enumerate(rewards):
        print("reward "+str(ind))
        print(reward)

if __name__ == "__main__":
    main()
