import unittest
from copy import deepcopy

from influence.testing import TestEnv

class TestDifference(TestEnv):
    """Run some simple checks to make sure D works properly
    - 1 rover, 1 POI. D is the same as G
    - 2 rovers, 1 POI. The closer rover gets G through D. The further rover gets rewarded 0
    - 2 rovers, 2 POIs. Each rover gets rewarded for its POI
    - 4 rovers, 4 POIs. Each rover gets rewarded for its POI
    - 1 rover, 1 uav, 1 POI. G is 1.0. Rover gets D=1.0, uav gets D=0.0
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_poi_config['observation_radius'] = 5.0

    def get_default_config(self):
        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 50.]
        return config
    
    def get_one_rover_one_poi_config(self):
        config = self.get_default_config()
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [10.0, 10.0]
        config['env']['agents']['rovers'] = [deepcopy(rover_config)]
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [10.0, 10.0]
        config['env']['pois']['rover_pois'] = [deepcopy(poi_config)]
        return config
    
    def get_one_rover_one_uav_one_poi_config(self):
        config = self.get_one_rover_one_poi_config()
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = [10.0, 10.0]
        config['env']['agents']['uavs'].append(deepcopy(uav_config))
        return config
    
    def get_two_rovers_one_poi_config(self):
        config = self.get_one_rover_one_poi_config()
        # Add a rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [40.0, 40.0]
        config['env']['agents']['rovers'].append(deepcopy(rover_config))
        return config

    def get_two_rovers_two_pois_config(self):
        config = self.get_two_rovers_one_poi_config()
        # Add a POI at the second rover
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [40.0, 40.0]
        config['env']['pois']['rover_pois'].append(deepcopy(poi_config))
        return config
    
    def get_four_rovers_four_pois_config(self):
        config = self.get_two_rovers_two_pois_config()
        # Add in 2 more rovers and 2 more pois for those rovers
        new_positions = [
            [10.0, 40.0],
            [40.0, 10.0]
        ]
        for position in new_positions:
            rover_config = self.get_default_rover_config()
            rover_config['position']['fixed'] = position
            config['env']['agents']['rovers'].append(deepcopy(rover_config))
            poi_config = self.get_default_poi_config()
            poi_config['position']['fixed'] = position
            config['env']['pois']['rover_pois'].append(deepcopy(poi_config))
        return config

    def test_one_rover_one_poi(self):
        # -- 1 rover, 1 POI. D is the same as G
        config = self.get_one_rover_one_poi_config()
        # Check with G
        self.assert_correct_rewards(config, expected_rewards=[1.0])
        # Check with D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0])

    def test_two_rovers_one_poi(self):
        # -- 2 rovers, 1 POI. The closer rover gets G through D. The further rover gets rewarded 0
        config = self.get_two_rovers_one_poi_config()
        # Check with G for both rovers
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
        # Check with D for both rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])
    
    def test_two_rovers_two_pois(self):
        # -- 2 rovers, 2 POIs. Each rover gets rewarded for its POI
        config = self.get_two_rovers_two_pois_config()
        # Check with G for both rovers
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0])
        # Check with D for both rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
    
    def test_four_rovers_four_pois(self):
        # -- 4 rovers, 4 POIs. Each rover gets rewarded for its POI
        config = self.get_four_rovers_four_pois_config()
        # Check G for all 4 rovers
        self.assert_correct_rewards(config, expected_rewards=[4.0]*4)
        # Switch rovers to D and check D for all 4 rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0]*4)

    def test_one_rover_one_uav_one_poi(self):
        config = self.get_one_rover_one_uav_one_poi_config()
        # Check G for rover and uav
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
        # Switch rover and uav to D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        # Check D for rover and uav
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])

if __name__ == '__main__':
    unittest.main()
