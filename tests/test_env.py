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
from influence.custom_env import createEnv

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
        def distance(entity_0, entity_1):
            return np.sqrt((entity_0.position().x - entity_1.position().x)**2 + (entity_0.position().y - entity_1.position().y)**2)
        def compute_poi_reward(poi, agent):
            return 1.0 / max(distance(poi, agent), 1.0)
        expected_G = compute_poi_reward(pois[0], agents[0]) + compute_poi_reward(pois[1], agents[1])
        for agent_reward in rewards:
            self.assertTrue(np.isclose(expected_G, agent_reward), f"{expected_G} != {agent_reward}")

    def test_G(self):
        """Check that G is computed as expected for different
        - poi observation radii
        - rover observation radii
        - coupling requirements
        - uavs in the environment (should not affect G)
        """
        pass
        # Put just one poi and one rover
#         rover_pos = [24.0, 25.0]
#         poi_pos = [24.0, 24.0]
#         config = {
#             'env': {
#                 'agents': {
#                     'rovers': [
#                         {
#                             'observation_radius': 1000.0,
#                             'position': {
#                                 'spawn_rule': 'fixed',
#                                 'fixed': agent_0_pos
#                             },
#                             'resolution': 90,
#                             'reward_type': 'Global'
#                         }
#                     ],
#                     'uavs': []
#                 },
#                 'pois': {
#                     'rover_pois': [
#                         {
#                             'coupling': 1,
#                             'observation_radius': 1000.0,
#                             'position': {
#                                 'spawn_rule': 'fixed',
#                                 'fixed': poi_0_pos
#                             },
#                             'value': 1.0
#                         }
#                     ],
#                     'hidden_pois': []
#                 },
#                 'map_size': [50., 50.]
#             }
#         }


if __name__ == '__main__':
    unittest.main()
