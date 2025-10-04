"""This is testing a new lidar for rovers where they sense their distance to all
uavs within their observation radius
"""

import unittest
import numpy as np
from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestUavDistanceLidar(TestEnv):
    def get_config_a(self):
        config = self.get_env_template_config()
        config['env']['map_size'] = [100., 100.]
        # Populate with 4 pois (rovers should not sense these)
        poi_positions = [
            [25., 75.],
            [50., 50.],
            [75., 75.]
        ]
        for position in poi_positions:
            poi_config = self.get_default_poi_config()
            poi_config['position']['fixed'] = position
            config['env']['pois']['hidden_pois'].append(poi_config)
        # Put 3 rovers
        # First two rovers are close to each other,
        # third rover is far away
        rover_positions = [
            [25., 25.],
            [25., 30.],
            [75., 25.]
        ]
        for position in rover_positions:
            rover_config = self.get_default_rover_config()
            rover_config['sensor'] = {
                'type': 'UavDistanceLidar'
            }
            rover_config['position']['fixed'] = position
            config['env']['agents']['rovers'].append(rover_config)
        # Put 5 uavs
        uav_positions = [
            [10., 10.],
            [10., 20.],
            [20., 20.],
            [80., 20.],
            [90., 90.]
        ]
        for position in uav_positions:
            uav_config = self.get_default_uav_config()
            uav_config['position']['fixed'] = position
            config['env']['agents']['uavs'].append(uav_config)
        return config

    def get_config_b(self):
        # Same as config_a but with a smaller observation radius for rovers
        config = self.get_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['observation_radius'] = 25.0
        return config

    def compute_expected_rover_observations(self, config):
        expected_rover_observations = []
        # Iterate through rovers and build out our expected observation for that rover
        for rover_config in config['env']['agents']['rovers']:
            rover_observation = []
            for uav_config in config['env']['agents']['uavs']:
                distance = np.linalg.norm(
                    np.array(rover_config['position']['fixed']) \
                    - np.array(uav_config['position']['fixed'])
                )
                if distance <= rover_config['observation_radius']:
                    rover_observation.append(
                        distance / rover_config['observation_radius']
                    )
                else:
                    rover_observation.append(-1)
            expected_rover_observations.append(rover_observation)
        return expected_rover_observations

    def test_a(self):
        """Test that distances are computed properly"""
        # Initialize environment
        config = self.get_config_a()
        env = createEnv(config)

        # Get the observations
        observations, _ = env.reset()

        # Create our expected observations for rovers
        expected_rover_observations = self.compute_expected_rover_observations(config)

        # Check expected observations against observations from the env
        for expected_observation, observation in zip(expected_rover_observations, observations):
            processed_obs = self.extract_observation(observation)
            self.assertTrue(len(expected_observation) == len(processed_obs))
            self.assert_close_lists(expected_observation, processed_obs)

    def test_b(self):
        """Test that uavs that are too far away are appropriately filtered out"""
        config = self.get_config_b()
        env = createEnv(config)

        # Get the observations
        observations, _ = env.reset()

        # Create our expected observations for rovers
        expected_rover_observations = self.compute_expected_rover_observations(config)

        # Check expected observations against observations from the env
        for expected_observation, observation in zip(expected_rover_observations, observations):
            processed_obs = self.extract_observation(observation)
            self.assertTrue(len(expected_observation) == len(processed_obs))
            self.assert_close_lists(expected_observation, processed_obs)


if __name__ == '__main__':
    unittest.main()
