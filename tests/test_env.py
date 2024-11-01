"""
Unit tests to check reward functions work how I expect them to

NOTE: TESTS ARE NOT EXTENSIVE!!! Bugs can sneak past them.
No guarantees of bug-free code just because the unit tests are happy

To run individual tests:
python tests/test_env.py TestEnv.test_env
or
python -m unittest tests.test_env.TestEnv.test_env

"""

import unittest
import numpy as np
from influence.librovers import rovers
from influence.custom_env import createEnv

# Helper functions
def distance(entity_0, entity_1):
    return np.sqrt((entity_0.position().x - entity_1.position().x)**2 + (entity_0.position().y - entity_1.position().y)**2)
def compute_poi_reward(poi, agent):
    return 1.0 / max(distance(poi, agent), 1.0)

class TestEnv(unittest.TestCase):
    def test_env(self):
        """Set up an environment and check everything runs properly"""
        agent_0_pos = [25.0, 25.0]
        agent_1_pos = [24.0, 24.0]
        agent_2_pos = [26.0, 24.0]
        poi_0_pos = [35.0, 40.0]
        poi_1_pos = [22.0, 15.0]
        config = {
            'env': {
                'agents': {
                    'rovers': [
                        {
                            'observation_radius': 1000.0,
                            'position': {
                                'spawn_rule': 'fixed',
                                'fixed': agent_0_pos
                            },
                            'resolution': 90,
                            'reward_type': 'Global'
                        },
                        {
                            'observation_radius': 1000.0,
                            'position': {
                                'spawn_rule': 'fixed',
                                'fixed': agent_1_pos
                            },
                            'resolution': 90,
                            'reward_type': 'Global'
                        }
                    ],
                    'uavs': [
                        {
                            'observation_radius': 1000.0,
                            'position': {
                                'spawn_rule': 'fixed',
                                'fixed': agent_2_pos
                            },
                            'resolution': 90,
                            'reward_type': 'Global'
                        }
                    ]
                },
                'pois': {
                    'rover_pois': [
                        {
                            'coupling': 1,
                            'observation_radius': 1000.0,
                            'position': {
                                'spawn_rule': 'fixed',
                                'fixed': poi_0_pos
                            },
                            'value': 1.0
                        }
                    ],
                    'hidden_pois': [
                        {
                            'coupling': 1,
                            'observation_radius': 1000.0,
                            'position': {
                                'spawn_rule': 'fixed',
                                'fixed': poi_1_pos
                                },
                            'value': 1.0
                        }
                    ]
                },
                'map_size': [50., 50.]
            }
        }

        # createEnv returns a rovers.Environment[rovers.CustomInit]
        env = createEnv(config)

        # Extract additional info we need to check
        agents = env.rovers()
        pois = env.pois()

        # Check we have the correct number of agents and pois
        expected_num_agents = 3
        expected_num_pois = 2
        self.assertTrue(expected_num_agents == len(agents))
        self.assertTrue(expected_num_pois == len(pois))

        # We need to reset the env to complete the setup
        observations, rewards = env.reset()

        # Check the positions of rovers, uavs, pois
        # -- agents
        self.assertTrue(np.isclose(agent_0_pos[0], agents[0].position().x))
        self.assertTrue(np.isclose(agent_0_pos[1], agents[0].position().y))
        self.assertTrue(np.isclose(agent_1_pos[0], agents[1].position().x))
        self.assertTrue(np.isclose(agent_1_pos[1], agents[1].position().y))
        self.assertTrue(np.isclose(agent_2_pos[0], agents[2].position().x))
        self.assertTrue(np.isclose(agent_2_pos[1], agents[2].position().y))
        # -- pois
        self.assertTrue(np.isclose(poi_0_pos[0], pois[0].position().x))
        self.assertTrue(np.isclose(poi_0_pos[1], pois[0].position().y))
        self.assertTrue(np.isclose(poi_1_pos[0], pois[1].position().x))
        self.assertTrue(np.isclose(poi_1_pos[1], pois[1].position().y))

        # Check the observations of rovers, uavs
        def sensor_value(entity_0, entity_1):
            squared_distance = (entity_0.position().x - entity_1.position().x)**2 + (entity_0.position().y - entity_1.position().y)**2
            return 1.0 / max([0.001, squared_distance])

        agent_0_sense_agent_1 = 0.5
        agent_0_sense_agent_2 = 0.5
        agent_0_sense_poi_0 = sensor_value(agents[0], pois[0])
        agent_1_sense_agent_0 = 0.5
        agent_1_sense_agent_2 = 0.25
        agent_1_sense_poi_0 = sensor_value(agents[1], pois[0])
        agent_2_sense_agent_0 = 0.5
        agent_2_sense_agent_1 = 0.25
        agent_2_sense_poi_0 = sensor_value(agents[2], pois[0])
        agent_2_sense_poi_1 = sensor_value(agents[2], pois[1])
        expected_observations = [
            # Agent 0 is a rover that senses the other rover, the uav, and the rover_poi
            [-1, -1, agent_0_sense_agent_1, -1, 
             -1, -1, -1, agent_0_sense_agent_2, 
             agent_0_sense_poi_0, -1, -1, -1],
            # Agent 1 is a rover that senses the first rover, the uav, and the rover_poi
            [agent_1_sense_agent_0, -1, -1, -1,
             agent_1_sense_agent_2, -1, -1, -1,
             agent_1_sense_poi_0, -1, -1, -1],
            # Agent 2 is a uav that senses the rovers, the rover_poi, and the hidden_poi
            [-1, agent_2_sense_agent_0, agent_2_sense_agent_1, -1,
             -1, -1, -1, -1,
              agent_2_sense_poi_0, -1, agent_2_sense_poi_1, -1]
        ]
        for expected_observation, observation in zip(expected_observations, observations):
            for i in range(observation.size()):
                self.assertTrue(np.isclose(expected_observation[i], observation[i,0]))

        # Check the rewards
        expected_G = compute_poi_reward(pois[0], agents[0]) + compute_poi_reward(pois[1], agents[1])
        for agent_reward in rewards:
            self.assertTrue(np.isclose(expected_G, agent_reward), f"{expected_G} != {agent_reward}")

    def test_G(self):
        """Check that G is computed as expected for different cases
        - additional rovers with low coupling (if coupling is satisfied, addtl rovers shouldn't change anything)
        - poi observation radii
        - uavs in the environment (should not affect G)
        - more pois
        """
        def compute_G(env):
            return env.rovers()[0].reward(rovers.AgentPack(0, env.rovers(), env.pois()))
        def get_default_rover_config():
            return {
                'observation_radius': 1000.0,
                'position': {
                    'spawn_rule': 'fixed',
                    'fixed': [25.0, 24.0]
                },
                'resolution': 90,
                'reward_type': 'Global'
            }
        def get_default_uav_config():
            return {
                'observation_radius': 1000.0,
                'position': {
                    'spawn_rule': 'fixed',
                    'fixed': [23.0, 24.0]
                },
                'resolution': 90,
                'reward_type': 'Global'
            }
        def get_default_poi_config():
            return {
                'coupling': 1,
                'observation_radius': 1000.0,
                'position': {
                    'spawn_rule': 'fixed',
                    'fixed': [24.0, 24.0]
                },
                'value': 1.0
            }
        def get_default_config():
            return {
                'env': {
                    'agents': {
                        'rovers': [
                            get_default_rover_config()
                        ],
                        'uavs': []
                    },
                    'pois': {
                        'rover_pois': [
                           get_default_poi_config() 
                        ],
                        'hidden_pois': []
                    },
                    'map_size': [50., 50.]
                }
            }
        # Check one rover, one poi. Reward of 1.0
        config = get_default_config()
        env = createEnv(config)
        _, (reward) = env.reset()
        expected_reward = 1.0
        self.assertTrue(np.isclose(reward, expected_reward))

        # Place the rover closer to the poi. Reward should still be 1.0
        # env.rovers()[0].set_position(24.5, 24.0)
        config['env']['agents']['rovers'][0]['position']['fixed'] = [24.5, 24.0]
        env = createEnv(config)
        _, (reward) = env.reset()
        expected_reward = 1.0
        self.assertTrue(np.isclose(reward, expected_reward))

        # Place the rover further away. Reward should be 0.5
        config['env']['agents']['rovers'][0]['position']['fixed'] = [26.0, 24.0]
        env = createEnv(config)
        _, (reward) = env.reset()
        expected_reward = 0.5
        self.assertTrue(np.isclose(reward, expected_reward))

        # Make the rover follow a path. 
        # Reward should be 0.5 based on final position in path to poi
        rover_path = [
            [24.5, 24.0],
            [26.0, 24.0]
        ]
        config = get_default_config()
        _, _ = env.reset()
        for position in rover_path:
            env.rovers()[0].set_position(position[0], position[1])
        reward = compute_G(env)
        expected_reward = 0.5
        self.assertTrue(np.isclose(reward, expected_reward), f'{reward} != {expected_reward}')

        # Now let's add another copy of our rover to make sure we are not double counting
        config['env']['agents']['rovers'].append(get_default_rover_config())
        env = createEnv(config)
        _, (reward, _) = env.reset()
        expected_reward = 1.0
        self.assertTrue(np.isclose(reward, expected_reward))

        # Add another 10 and check again for good measure
        for _ in range(10):
            config['env']['agents']['rovers'].append(get_default_rover_config())
        env = createEnv(config)
        _, rewards = env.reset()
        reward = rewards[0]
        expected_reward = 1.0
        self.assertTrue(np.isclose(reward, expected_reward))

        # Reset the config, and bring down the observation radius for the poi
        # G should still be 1.0
        config = get_default_config()
        config['env']['pois']['rover_pois'][0]['observation_radius'] = 5.0
        env = createEnv(config)
        _, (reward) = env.reset()
        expected_reward = 1.0
        self.assertTrue(np.isclose(reward, expected_reward))
        
        # But if the rover moves away, G should go down
        env.rovers()[0].set_position(27.0, 24.0)
        reward = compute_G(env)
        expected_reward = compute_poi_reward(env.pois()[0], env.rovers()[0])
        self.assertTrue(np.isclose(reward, expected_reward))

        # If the rover moves far enough away, G becomes 0.0
        env.rovers()[0].set_position(30.0, 24.0)
        reward = compute_G(env)
        expected_reward = 0.0
        self.assertTrue(np.isclose(reward, expected_reward))

        # Let's set the default config and throw in some uavs. Should not change G
        config = get_default_config()
        expected_reward = 1.0
        for _ in range(3):
            config['env']['agents']['uavs'].append(get_default_uav_config())
            env = createEnv(config)
            _, rewards = env.reset()
            reward = rewards[0]
            self.assertTrue(np.isclose(reward, expected_reward))
        
        # Let's set the default config and add more pois.
        config = get_default_config()
        more_poi_positions = [
            [25.0, 35.0],
            [35.0, 35.0]
        ]
        poi_config = get_default_poi_config()
        for position in more_poi_positions:
            poi_config['position']['fixed'] = position
            config['env']['pois']['rover_pois'].append(poi_config)
        # With only 1 rover, G should be 1.0 + the value of the other two pois
        env = createEnv(config)
        reward = compute_G(env)
        expected_reward = 1.0 + \
            compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assertTrue(np.isclose(reward, expected_reward))

        # Let's add uavs to observe the remaining pois 
        # (reward should not change because uavs can't observe pois)
        uav_config = get_default_uav_config()
        for position in more_poi_positions:
            uav_config['position']['fixed'] = position
        env = createEnv(config)
        reward = compute_G(env)
        expected_reward = 1.0 + \
            compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assertTrue(np.isclose(reward, expected_reward))

        # Now let's add rovers to actually observe the remaining pois
        rover_config = get_default_rover_config()
        for position in more_poi_positions:
            rover_config['position']['fixed'] = position
        env = createEnv(config)
        reward = compute_G(env)
        expected_reward = 3.0
        self.assertTrue(np.isclose(reward, expected_reward))

if __name__ == '__main__':
    unittest.main()
