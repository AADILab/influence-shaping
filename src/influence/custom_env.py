from influence.librovers import rovers, thyme
import numpy as np
import cppyy
import random

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

        # print("Observe Agents")
        # Observe agents
        for i in range(agent_pack.agents.size()):
            # print("Sensing agent")
            # Do not observe yourself
            if i == agent_pack.agent_index:
                # print("Nope, that one is me")
                continue
            # Get angle and distance to sensed agent
            sensed_agent = agent_pack.agents[i]
            angle = calculateAngle(agent.position(), sensed_agent.position())
            distance = calculateDistance(agent.position(), sensed_agent.position())
            # print("angle: ", angle)
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
            # Bin the agent according to type
            if self.agent_types[i] == "rover":
                rover_values[sector].append(1.0 / max([0.001, distance**2]))
            elif self.agent_types[i] == "uav":
                uav_values[sector].append(1.0 / max([0.001, distance**2]))

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
    if reward_type == "Global":
        Reward = rovers.rewards.Global
    elif reward_type == "Difference":
        Reward = rovers.rewards.Difference
    rover = rovers.Rover[SmartLidar, Discrete, Reward](obs_radius, SmartLidar(resolution=resolution, composition_policy=rovers.Density(), agent_types=agent_types, poi_types=poi_types), Reward())
    rover.type = "rover"
    return rover

# Now create a UAV
def createUAV(obs_radius, reward_type, resolution, agent_types, poi_types):
    Discrete = thyme.spaces.Discrete
    if reward_type == "Global":
        Reward = rovers.rewards.Global
    elif reward_type == "Difference":
        Reward = rovers.rewards.Difference
    uav = rovers.Rover[SmartLidar, Discrete, Reward](obs_radius, SmartLidar(resolution=resolution, composition_policy=rovers.Density(), agent_types=agent_types, poi_types=poi_types), Reward())
    uav.type = "uav"
    return uav

# Now create a POI constraint where this POI can only be observed by rovers with "rover" type
class RoverConstraint(rovers.IConstraint):
    def __init__(self, coupling, is_rover_list):
        super().__init__()
        self.coupling = coupling
        self.is_rover_list = is_rover_list

    def is_satisfied(self, entity_pack):
        count = 0
        dists = []
        constraint_satisfied = False
        for is_rover, agent in zip(self.is_rover_list, entity_pack.agents):
            if is_rover:
                dist = calculateDistance(agent.position(), entity_pack.entity.position())
                dists.append(dist)
                if dist <= agent.obs_radius() and dist <= entity_pack.entity.obs_radius():
                    count += 1
                    if count >= self.coupling:
                        constraint_satisfied = True
        if constraint_satisfied:
            dists.sort()
            dists = [max(1.0, dist) for dist in dists]
            constraint_value = float(self.coupling)
            for dist in dists:
                constraint_value = constraint_value*1.0/dist
            # print("constraint_value: ", constraint_value)
            return constraint_value
        return 0.0

def createRoverPOI(value, obs_rad, coupling, is_rover_list):
    roverConstraint = RoverConstraint(coupling, is_rover_list)
    poi = rovers.POI[RoverConstraint](value, obs_rad, roverConstraint)
    return poi

# This is just to help me track which POIs are nominally hidden from rovers
def createHiddenPOI(value, obs_rad, coupling, is_rover_list):
    return createRoverPOI(value, obs_rad, coupling, is_rover_list)

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
    NUM_UAVS = len(config["env"]["agents"]["rovers"])
    NUM_ROVER_POIS = len(config["env"]["pois"]["rover_pois"])
    NUM_HIDDEN_POIS = len(config["env"]["pois"]["hidden_pois"])

    agent_types = ["rover"]*NUM_ROVERS + ["uav"]*NUM_UAVS
    poi_types = ["rover"]*NUM_ROVER_POIS+["hidden"]*NUM_HIDDEN_POIS

    rovers_ = [
        createRover(
            obs_radius=rover["observation_radius"],
            reward_type=rover["reward_type"],
            resolution=rover["resolution"],
            agent_types=agent_types,
            poi_types=poi_types
        )
        for rover in config["env"]["agents"]["rovers"]
    ]
    uavs = [
        createUAV(
            obs_radius=uav["observation_radius"],
            reward_type=uav["reward_type"],
            resolution=uav["resolution"],
            agent_types=agent_types,
            poi_types=poi_types
        )
        for uav in config["env"]["agents"]["uavs"]
    ]
    agents = rovers_+uavs

    is_rover_list = [True if str_=="rover" else False for str_ in agent_types]

    rover_pois = [
        createRoverPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            coupling=poi["coupling"],
            is_rover_list=is_rover_list
        )
        for poi in config["env"]["pois"]["rover_pois"]
    ]
    hidden_pois = [
        createHiddenPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            coupling=poi["coupling"],
            is_rover_list=is_rover_list
        )
        for poi in config["env"]["pois"]["hidden_pois"]
    ]
    pois = rover_pois + hidden_pois

    env = Env(
        rovers.CustomInit(agent_positions, poi_positions),
        agents,
        pois,
        width=cppyy.gbl.ulong(config["env"]["map_size"][0]),
        height=cppyy.gbl.ulong(config["env"]["map_size"][1])
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
