"""There are going to be different types for hidden pois.
Each poi type requires a uav with the corresponding sensor to be able to sense that poi.
"""

import unittest
from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestPoiTypeSensor(TestEnv):
    def get_config_a(self):
        config = self.get_env_template_config()
        config['env']['map_size'] = [5., 5.]
        # Populate with 4 pois of unique types
        poi_positions = [
            [1., 1.],
            [1., 2.],
            [2., 1.],
            [2., 2.]
        ]
        poi_types = ['A', 'B', 'C', '']
        for position, type_ in zip(poi_positions, poi_types):
            poi_config = self.get_default_poi_config()
            poi_config['position']['fixed'] = position
            poi_config['subtype'] = type_
            config['env']['pois']['hidden_pois'].append(poi_config)
        # Populate with 3 uavs each capable of observing a different poi type
        # (But each one can observe the None-type poi. No type means anyone can observe it)
        uav_positions = [
            [1., 1.],
            [1., 2.],
            [2., 1.]
        ]
        uav_observable_types = [
            ['A', 'B'],
            ['A', 'C'],
            ['C']
        ]
        for position, obs_types in zip(uav_positions, uav_observable_types):
            uav_config = self.get_default_uav_config()
            uav_config['position']['fixed'] = position
            uav_config['observable_subtypes'] = obs_types
            uav_config['sensor']={'accum_type': 'sum'}
            config['env']['agents']['uavs'].append(uav_config)
        return config

    def test_a(self):
        # Initialize environment with uavs and pois
        config = self.get_config_a()
        env = createEnv(config)

        # Get the observations
        observations, _ = env.reset()

        # Create our expected observations
        expected_observations = [
            # Uav 0 should see the other uavs, and the pois at [1,1], [2,2], [2,1]
            [-1, -1, -1, -1,  1,  1, -1, -1, 1000.5, 1, -1, -1],
            # Uav 1 should see the other uavs and the pois at [1,1], [2,1], [2,2]
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.5, 1.0, -1.0, -1.0, 1.5],
            # Uav 2 should see the other uavs and the pois at [2,1], [2,2]
            [-1.0, -1.0, -1.0, -1.0, -1.0, 0.4999999999999999, 1.0, -1.0, 1000.0, 1.0, -1.0, -1.0]
        ]

        # Process the observations for easier debugging
        processed_observations = [[observation[i,0] for i in range(12)] for observation in observations]

        # Now check the observations of each uav
        for expected_observation, actual_observation in zip(expected_observations, processed_observations):
            self.assert_close_lists(expected_observation, actual_observation)

if __name__ == '__main__':
    unittest.main()
