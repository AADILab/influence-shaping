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

    def get_two_rovers_two_uavs_two_pois_config(self):
        config = self.get_one_rover_one_uav_one_poi_config()
        # Add a second poi, rover, and uav to the same location
        config['env']['agents']['rovers'].append(deepcopy(config['env']['agents']['rovers'][0]))
        config['env']['agents']['uavs'].append(deepcopy(config['env']['agents']['uavs'][0]))
        config['env']['pois']['rover_pois'].append(deepcopy(config['env']['pois']['rover_pois'][0]))
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

    def test_two_rovers_two_uavs_two_pois_a(self):
        # Both rovers and both uavs on top of both pois
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Check G for all agents
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0, 2.0, 2.0])
        # Check D for all agents. (0.0 for each rover because the other rover is in the same place. 0.0 for uavs because they cannot observe pois)
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0, 0.0, 0.0])
    
    def test_two_rovers_two_uavs_two_pois_b(self):
        # Both rovers at one poi. Both uavs at another poi
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Move the second POI
        config['env']['pois']['rover_pois'][1]['position']['fixed'] = [40.0, 10.0]
        # Move the uavs to the second POI
        for uav_config in config['env']['agents']['uavs']:
            uav_config['position']['fixed'] = [40.0, 10.0]
        # Check G for all agents. Should be 1.0
        G = 1.0
        self.assert_correct_rewards(config, expected_rewards=[G, G, G, G])
        # Now check D. D should be 0.0 for rovers because if you remove either one, the other one is there at the same spot
        # D should also be 0.0 for uavs because G never changes when uavs are removed
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0, 0.0, 0.0])
    
    def test_two_rovers_two_uavs_two_pois_c(self):
        # One rover/uav pair at each poi
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Move the second rover, uav, and poi to different location
        config['env']['pois']['rover_pois'][1]['position']['fixed'] = [40.0, 10.0]
        config['env']['agents']['rovers'][1]['position']['fixed'] = [40.0, 10.0]
        config['env']['agents']['uavs'][1]['position']['fixed'] = [40.0, 10.0]
        # Check G for all agents. Should 2.0 because both pois are observed
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0, 2.0, 2.0])
        # Now check D. Should be (1.0 - impact of other rover) for each rover and 0.0 for each uav
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0, 0.0, 0.0])

if __name__ == '__main__':
    unittest.main()
