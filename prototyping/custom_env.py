from librovers import rovers, thyme
import numpy as np
import cppyy

def calculateDistance(position_0, position_1):
    return np.linalg.norm([position_0.x-position_1.x, position_0.y-position_1.y])

def calculateAngle(position_0, position_1):
    pos0 = np.array([position_0.x, position_0.y])
    pos1 = np.array([position_1.x, position_1.y])
    # Create a vector from position 0 to 1
    vec = pos1 - pos0
    # Take the arctan2 of the y, x of that vector
    return np.arctan2(vec[1], vec[0]) * 180./np.pi

# Agents should be able to distinguish between rovers and uavs...
# Unfortunately this means writing my own Lidar class that can differentiate between agent types
class SmartLidar(rovers.Lidar[rovers.Density]):
    def __init__(self, resolution, composition_policy, agent_types):
        super().__init__(resolution, composition_policy)
        self.agent_types = agent_types
        self.m_resolution = resolution
        self.m_composition = composition_policy

    def scan(self, agent_pack):
        num_sectors = int(360. / self.m_resolution)
        poi_values = [[] for _ in range(num_sectors)]
        rover_values = [[] for _ in range(num_sectors)]
        uav_values = [[] for _ in range(num_sectors)]
        agent = agent_pack.agents[agent_pack.agent_index]

        # Observe POIs
        print("Observe POIs")
        for sensed_poi in agent_pack.entities:
            print("Sensing POI")
            # Get angle and distance to POI
            angle = calculateAngle(agent.position(), sensed_poi.position())
            distance = calculateDistance(agent.position(), sensed_poi.position())
            print("angle: ", angle)
            print("distance: ", distance)
            # Skip this POI if it is out of range
            if distance > agent.obs_radius():
                print("continue, out of range")
                continue
            # Bin the POI according to where it was sensed
            if angle < 360.0:
                sector = int( angle / self.m_resolution )
            else:
                sector = 0
            print("sector: ", sector, type(sector))
            poi_values[sector].append(sensed_poi.value() / max([0.001, distance**2]))

        print("Observe Agents")
        # Observe agents
        for i in range(agent_pack.agents.size()):
            print("Sensing agent")
            # Do not observe yourself
            if i == agent_pack.agent_index:
                print("Nope, that one is me")
                continue
            # Get angle and distance to sensed agent
            sensed_agent = agent_pack.agents[i]
            angle = calculateAngle(agent.position(), sensed_agent.position())
            distance = calculateDistance(agent.position(), sensed_agent.position())
            print("angle: ", angle)
            print("distance: ", distance)
            # Skip the agent if the sensed agent is out of range
            if distance > agent.obs_radius():
                print("continue, out of range")
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

        print("rover_values: ", rover_values)
        print("uav_values: ", uav_values)
        print("poi_values: ", poi_values)

        # Encode the state
        print("Encoding state")
        state = np.array([-1.0 for _ in range(num_sectors*3)])
        print("state: ", state)
        for i in range(num_sectors):
            print("Building sector ", i)
            num_rovers = len(rover_values[i])
            num_uavs = len(uav_values[i])
            num_pois = len(poi_values[i])

            if num_rovers > 0:
                print("num_rovers > 0")
                print("rover_values["+str(i)+"]: ", rover_values[i], type(rover_values[i]), type(rover_values[i][0]))
                print("num_rovers: ", type(num_rovers))
                state[i] = self.m_composition.compose(rover_values[i], 0.0, num_rovers)
            if num_uavs > 0:
                print("num_uavs > 0")
                print("uav_values["+str(i)+"]: ", uav_values[i], type(uav_values[i]), type(uav_values[i][0]))
                print("num_uavs: ", type(num_uavs))
                state[num_sectors + i] = self.m_composition.compose(uav_values[i], 0.0, num_uavs)
            if num_pois > 0:
                print("num_pois > 0")
                print("poi_values["+str(i)+"]: ", poi_values[i], type(poi_values[i]), type(poi_values[i][0]))
                print("num_pois: ", type(num_pois))
                # Convert poi_values[i] to a std::vector<double> to satisfy cppyy
                # Not sure why this is necessary for poi_values but not rover_values or uav_values
                cpp_vector = cppyy.gbl.std.vector[cppyy.gbl.double]()
                for p in poi_values[i]:
                    cpp_vector.push_back(p)
                state[num_sectors*2 + i] = self.m_composition.compose(cpp_vector, 0.0, num_pois)

        return rovers.tensor(state)

# First we're going to create a simple rover
def createRover(obs_radius, reward_type = "Global"):
    Discrete = thyme.spaces.Discrete
    if reward_type == "Global":
        Reward = rovers.rewards.Global
    elif reward_type == "Difference":
        Reward = rovers.rewards.Difference
    rover = rovers.Rover[SmartLidar, Discrete, Reward](obs_radius, SmartLidar(resolution=90, composition_policy=rovers.Density(), agent_types=["rover", "rover", "uav", "uav"]), Reward())
    rover.type = "rover"
    return rover

# Now create a UAV
def createUAV(obs_radius, reward_type = "Global"):
    Discrete = thyme.spaces.Discrete
    if reward_type == "Global":
        Reward = rovers.rewards.Global
    elif reward_type == "Difference":
        Reward = rovers.rewards.Difference
    uav = rovers.Rover[SmartLidar, Discrete, Reward](obs_radius, SmartLidar(resolution=30, composition_policy=rovers.Density(), agent_types=["rover", "rover", "uav", "uav"]), Reward())
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
        for is_rover, agent in zip(self.is_rover_list, entity_pack.agents):
            if is_rover:
                dist = calculateDistance(agent.position(), entity_pack.entity.position())
                if dist <= agent.obs_radius() and dist <= entity_pack.entity.obs_radius():
                    count += 1
                    if count >= self.coupling:
                        return True
        return False

def createRoverPOI(value, obs_rad, coupling, is_rover_list):
    roverConstraint = RoverConstraint(coupling, is_rover_list)
    poi = rovers.POI[RoverConstraint](value, obs_rad, roverConstraint)
    return poi

# Running into errors setting up the environment
# Let's try it with regular POIs
def createPOI(value, obs_rad, coupling, is_rover_list):
    countConstraint = rovers.CountConstraint(coupling)
    poi = rovers.POI[rovers.CountConstraint](value, obs_rad, countConstraint)
    return poi

# Alright let's give this a try
def main():
    Env = rovers.Environment[rovers.CustomInit]
    agent_positions = [
        [5. , 5.],
        [3. , 3.],
        # [1. , 1.],
        # [9. , 9.]
    ]
    poi_positions = [
        [1. , 9.],
        [9. , 9.]
    ]
    rover_obs_rad = 100.
    uav_obs_rad = 100.
    agents = [
        createRover(rover_obs_rad, reward_type = "Difference"),
        createRover(rover_obs_rad, reward_type = "Difference"),
        # createUAV(uav_obs_rad, reward_type = "Difference"),
        # createUAV(uav_obs_rad, reward_type = "Difference")
    ]
    pois = [
        createRoverPOI(value=1. , obs_rad=1. , coupling=1, is_rover_list=[True, True, False, False]),
        createRoverPOI(value=1. , obs_rad=1. , coupling=1, is_rover_list=[True, True, False, False])
    ]
    env = Env(rovers.CustomInit(agent_positions, poi_positions), agents, pois)

    states, rewards = env.reset()

    print("States:")
    for state in states:
        print(state.transpose())

    print("Rewards:")
    for reward in rewards:
        print(reward)

if __name__ == "__main__":
    main()
