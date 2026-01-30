from typing import List
from influence.librovers import rover_domain
import numpy as np
import cppyy
import random
from pprint import pprint

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

def createAgent(agent_config, poi_subtypes, agent_observable_subtypes, accum_type, measurement_type, type_, observation_radii, default_values, map_size):
    """Create an agent using the agent's config and type"""
    # unpack config
    reward_type = agent_config['reward_type']
    obs_radius = agent_config['observation_radius']
    resolution = agent_config['resolution']

    # Figure out what sensor type this agent is using
    sensor_type = 'SmartLidar'
    # pprint(agent_config)
    if 'sensor' in agent_config and 'type' in agent_config['sensor']:
        sensor_type = agent_config['sensor']['type']
    # print(sensor_type)

    # repackage indirect difference parameters
    IndirectDifferenceParameters = rover_domain.IndirectDifferenceParameters
    AutomaticParameters = rover_domain.AutomaticParameters
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
            ),
            add_G = indirect_difference_config['add_G'] if 'add_G' in indirect_difference_config else False
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
            ),
            add_G = False
        )

    # Set up agent bounds if they are not specified
    bounds = {
        'high_x': map_size[0],
        'low_x': 0.0,
        'high_y': map_size[1],
        'low_y': 0.0
    }
    if 'bounds' in agent_config:
        bounds = agent_config['bounds']

    Bounds = rover_domain.Bounds

    if sensor_type == 'SmartLidar':
        # Convert Python lists to C++ vectors for SmartLidar
        cpp_poi_subtypes = cppyy.gbl.std.vector[cppyy.gbl.std.string](poi_subtypes)

        # Convert nested list to C++ vector of vectors
        cpp_agent_observable_subtypes = cppyy.gbl.std.vector['std::vector<std::string>']()
        for subtype_list in agent_observable_subtypes:
            cpp_subtype_vec = cppyy.gbl.std.vector[cppyy.gbl.std.string](subtype_list)
            cpp_agent_observable_subtypes.push_back(cpp_subtype_vec)

        cpp_accum_type = cppyy.gbl.std.vector[cppyy.gbl.std.string](accum_type)
        cpp_measurement_type = cppyy.gbl.std.vector[cppyy.gbl.std.string](measurement_type)
        cpp_observation_radii = cppyy.gbl.std.vector[cppyy.gbl.double](observation_radii)
        cpp_default_values = cppyy.gbl.std.vector[cppyy.gbl.double](default_values)

        return rover_domain.DefaultAgent[rover_domain.SmartLidar[rover_domain.Density]](
            Bounds(
                low_x=bounds['low_x'],
                high_x=bounds['high_x'],
                low_y=bounds['low_y'],
                high_y=bounds['high_y']
            ),
            indirect_difference_parameters,
            reward_type,
            rover_domain.agent_type_from_string(type_),
            obs_radius,
            rover_domain.SmartLidar[rover_domain.Density](
                resolution,
                rover_domain.Density(),
                cpp_poi_subtypes,
                cpp_agent_observable_subtypes,
                cpp_accum_type,
                cpp_measurement_type,
                cpp_observation_radii,
                cpp_default_values
            )
        )
    elif sensor_type == 'RoverLidar':
        # Convert Python lists to C++ vectors for RoverLidar (no POI-related parameters)
        cpp_accum_type = cppyy.gbl.std.vector[cppyy.gbl.std.string](accum_type)
        cpp_measurement_type = cppyy.gbl.std.vector[cppyy.gbl.std.string](measurement_type)
        cpp_observation_radii = cppyy.gbl.std.vector[cppyy.gbl.double](observation_radii)
        cpp_default_values = cppyy.gbl.std.vector[cppyy.gbl.double](default_values)

        return rover_domain.DefaultAgent[rover_domain.RoverLidar[rover_domain.Density]](
            Bounds(
                low_x=bounds['low_x'],
                high_x=bounds['high_x'],
                low_y=bounds['low_y'],
                high_y=bounds['high_y']
            ),
            indirect_difference_parameters,
            reward_type,
            rover_domain.agent_type_from_string(type_),
            obs_radius,
            rover_domain.RoverLidar[rover_domain.Density](
                resolution,
                rover_domain.Density(),
                cpp_accum_type,
                cpp_measurement_type,
                cpp_observation_radii,
                cpp_default_values
            )
        )
    elif sensor_type == 'UavDistanceLidar':
        return rover_domain.DefaultAgent[rover_domain.UavDistanceLidar](
            Bounds(
                low_x=bounds['low_x'],
                high_x=bounds['high_x'],
                low_y=bounds['low_y'],
                high_y=bounds['high_y']
            ),
            indirect_difference_parameters,
            reward_type,
            rover_domain.agent_type_from_string(type_),
            obs_radius,
            rover_domain.UavDistanceLidar()
        )
    else:
        raise ValueError(f"Unknown sensor_type '{sensor_type}'.")

def createPOI(value, obs_rad, capture_radius, coupling, objective, hidden, disappears):
    if objective == 'sequential':
        objective_type = rover_domain.RoverSequenceObjective

    elif objective == 'final':
        objective_type = rover_domain.RoverObjective

    cpp_objective = objective_type(coupling)

    scope = rover_domain.VisibilityScope.ALL
    if hidden:
        scope = rover_domain.VisibilityScope.UAV_ONLY

    poi = rover_domain.DefaultPOI[objective_type](
        value,
        obs_rad,
        capture_radius,
        disappears,
        scope,
        cpp_objective
    )

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
    Env = rover_domain.Environment[rover_domain.CustomInit]

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

    for poi_config in config['env']['pois']['rover_pois']+config['env']['pois']['hidden_pois']:
        if 'disappear_bool' not in poi_config:
            poi_config['disappear_bool'] = True

    poi_subtypes = []
    for poi_config in config['env']['pois']['rover_pois']+config['env']['pois']['hidden_pois']:
        if 'subtype' in poi_config:
            poi_subtypes.append(poi_config['subtype'])
        else:
            poi_subtypes.append('')

    agent_observable_subtypes = []
    for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
        if 'observable_subtypes' in agent_config:
            agent_observable_subtypes.append(agent_config['observable_subtypes'])
        else:
            agent_observable_subtypes.append([])

    accum_type = []
    for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
        if 'sensor' in agent_config and 'accum_type' in agent_config['sensor']:
            accum_type.append(agent_config['sensor']['accum_type'])
        else:
            accum_type.append('average')

    measurement_type = []
    for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
        if 'sensor' in agent_config and 'measurement_type' in agent_config['sensor']:
            measurement_type.append(agent_config['sensor']['measurement_type'])
        else:
            measurement_type.append('inverse_distance_squared')

    observation_radii = [agent_config['observation_radius'] for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']]

    default_values = []
    for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
        if 'sensor' in agent_config and 'default_value' in agent_config['sensor']:
            default_values.append(agent_config['sensor']['default_value'])
        else:
            default_values.append(-1.0)

    rovers_ = [
        createAgent(
            agent_config=rover_config,
            poi_subtypes=poi_subtypes,
            agent_observable_subtypes=agent_observable_subtypes,
            accum_type=accum_type,
            measurement_type=measurement_type,
            type_='rover',
            observation_radii=observation_radii,
            default_values=default_values,
            map_size=config['env']['map_size']
        )
        for rover_config in config['env']['agents']['rovers']
    ]
    uavs = [
        createAgent(
            agent_config=uav_config,
            poi_subtypes=poi_subtypes,
            agent_observable_subtypes=agent_observable_subtypes,
            accum_type=accum_type,
            measurement_type=measurement_type,
            type_ = 'uav',
            observation_radii=observation_radii,
            default_values=default_values,
            map_size=config['env']['map_size']
        )
        for uav_config in config['env']['agents']['uavs']
    ]
    agents = rovers_+uavs

    # Fill defaults for new features
    for poi_config in config['env']['pois']['rover_pois']+config['env']['pois']['hidden_pois']:
        if 'constraint' not in poi_config:
            poi_config['constraint'] = 'final'
        if 'capture_radius' not in poi_config:
            poi_config['capture_radius'] = -1.0

    rover_pois = [
        createPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            capture_radius=poi["capture_radius"],
            coupling=poi["coupling"],
            objective=poi['constraint'],
            disappears=poi['disappear_bool'],
            hidden=False
        )
        for poi in config["env"]["pois"]["rover_pois"]
    ]
    hidden_pois = [
        createPOI(
            value=poi["value"],
            obs_rad=poi["observation_radius"],
            capture_radius=poi["capture_radius"],
            coupling=poi["coupling"],
            objective=poi['constraint'],
            disappears=poi['disappear_bool'],
            hidden=True
        )
        for poi in config["env"]["pois"]["hidden_pois"]
    ]
    pois = rover_pois + hidden_pois

    debug_reward_equals_G = False
    if 'debug' in config and 'reward_equals_G' in config['debug']:
        debug_reward_equals_G = config['debug']['reward_equals_G']

    env = Env(
        rover_domain.CustomInit(agent_positions, poi_positions),
        agents,
        pois,
        int(config["env"]["map_size"][0]),
        int(config["env"]["map_size"][1]),
        debug_reward_equals_G
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
