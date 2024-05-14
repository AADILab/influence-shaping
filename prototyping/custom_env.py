from librovers import rovers, thyme
import numpy as np

# First we're going to create a simple rover
def createRover(obs_radius):
    Dense = rovers.Lidar[rovers.Density]
    Discrete = thyme.spaces.Discrete
    Difference = rovers.rewards.Difference
    rover = rovers.Rover[Dense, Discrete, Difference](obs_radius, Dense(90), Difference())
    rover.type = "rover"
    return rover

# Now create a UAV
def createUAV(obs_radius):
    Dense = rovers.Lidar[rovers.Density]
    Discrete = thyme.spaces.Discrete
    Difference = rovers.rewards.Difference
    uav = rovers.Rover[Dense, Discrete, Difference](obs_radius, Dense(30), Difference())
    uav.type = "uav"
    return uav

def calculateDistance(position_0, position_1):
    return np.linalg.norm([position_0.x-position_1.x, position_0.y-position_1.y])

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
        [9. , 1.],
        [9. , 9.],
        [1. , 9.],
        [1. , 9.]
    ]
    poi_positions = [
        [1. , 9.],
        [9. , 9.]
    ]
    rover_obs_rad = 1.
    uav_obs_rad = 100.
    agents = [
        createRover(rover_obs_rad),
        createRover(rover_obs_rad),
        createUAV(uav_obs_rad),
        createUAV(uav_obs_rad)
    ]
    pois = [
        createPOI(value=1. , obs_rad=1. , coupling=1, is_rover_list=[True, True, False, False]),
        createPOI(value=1. , obs_rad=1. , coupling=1, is_rover_list=[True, True, False, False])
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
