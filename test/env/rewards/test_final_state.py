import unittest
from copy import deepcopy

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestFinalState(TestEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_poi_config['position']['fixed'] = [25.0, 25.0]
        self.default_poi_config['observation_radius'] = 5.0
        self.default_rover_config['position']['fixed'] = [25.0, 25.0]
        self.default_uav_config['position']['fixed'] = [25.0, 25.0]

    def get_default_config(self):
        config = self.get_env_template_config()
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        config['env']['agents']['uavs'].append(self.get_default_uav_config())
        config['env']['pois']['hidden_pois'].append(self.get_default_poi_config())
        config['env']['map_size'] = [50.0, 50.0]
        return config

    def get_two_rovers_two_uavs_two_pois_config(self):
        config = self.get_default_config()
        # Move first pair and poi to 10,10
        config['env']['agents']['rovers'][0]['position']['fixed'] = [10.0, 10.0]
        config['env']['agents']['uavs'][0]['position']['fixed'] = [10.0, 10.0]
        config['env']['pois']['hidden_pois'][0]['position']['fixed'] = [10.0, 10.0]
        # Add second pair and a poi at 40,40
        config['env']['agents']['rovers'].append(deepcopy(config['env']['agents']['rovers'][0]))
        config['env']['agents']['rovers'][1]['position']['fixed'] = [40.0, 40.0]
        config['env']['agents']['uavs'].append(deepcopy(config['env']['agents']['uavs'][0]))
        config['env']['agents']['uavs'][1]['position']['fixed'] = [40.0, 40.0]
        config['env']['pois']['hidden_pois'].append(deepcopy(config['env']['pois']['hidden_pois'][0]))
        config['env']['pois']['hidden_pois'][1]['position']['fixed'] = [40.0, 40.0]
        return config

    def get_two_rovers_two_uavs_two_pois_path_config(self):
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Move rovers to a spawn point near the center of the map
        for rover_config in config['env']['agents']['rovers']:
            rover_config['position']['fixed'] = [25.0, 25.0]
        for uav_config in config['env']['agents']['uavs']:
            uav_config['position']['fixed'] = [26.0, 24.0]
        # Move each poi to a different corner of the map. Bottom left and top left
        config['env']['pois']['hidden_pois'][0]['position']['fixed'] = [5.0, 45.0]
        config['env']['pois']['hidden_pois'][1]['position']['fixed'] = [5.0, 5.0]
        return config

class TestTwoRoversTwoUavsTwoPois(TestFinalState):
    def test_a(self):
        '''Each rover and uav pair are on a poi
        G should be 2.0
        D should be 1.0 for rovers
        D-Indirect should be 1.0 for uavs (and 1.0 for rovers)
        '''
        # Get our config
        config = self.get_two_rovers_two_uavs_two_pois_config()

        # First checking G
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0, 2.0, 2.0])
        # Switch everyone to D and check
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0, 0.0, 0.0])
        # Switch everyone to D-Indirect and check
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0, 1.0, 1.0])
        # Switch just rovers back to D and check
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0, 1.0, 1.0])

    def test_b(self):
        '''Both rovers and uavs are at both pois
        (All agents and pois at the same location)
        G should be 2.0
        D should be 0.0 for everyone
        D-Indirect will give all credit to the first uav (first uav gets credit for influencing BOTH rovers)
            so 0.0 for everyone except 2.0 for the first uav. This is default influence assignment behavior
        '''
        # Get the config
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Move second rover, uav, and poi to the same location as the first
        for entity_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']+config['env']['pois']['hidden_pois']:
            entity_config['position']['fixed'] = [10.0, 10.0]
        # Check G
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0, 2.0, 2.0])
        # Switch to D and check
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0, 0.0, 0.0])
        # Switch to D-Indirect and check
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0, 2.0, 0.0])

    def test_c(self):
        '''The first rover, uav pair are on the first poi
        The second rover and uav are scattered in the map
        The second poi is left unobserved
        G is 1.0
        D is 1.0 for the first rover, 0.0 for second rover (and uavs)
        D-Indirect is 1.0 for first rover and uav, 0.0 for second rover and uav
        '''
        # Get the config
        config = self.get_two_rovers_two_uavs_two_pois_config()
        # Move second rover and uav away from the poi
        config['env']['agents']['rovers'][1]['position']['fixed'] = [10.0, 40.0]
        config['env']['agents']['uavs'][1]['position']['fixed'] = [40.0, 10.0]
        # Check G
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0, 1.0, 1.0])
        # Check D
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0, 0.0, 0.0])
        # Check D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0, 1.0, 0.0])

class TestOneRoverOneUavOnePoi(TestFinalState):
    def test_a_Global(self):
        '''Rover and uav are directly on the POI. G should be 1.0
        D should be 1.0 for the rover, and 0.0 for the uav
        D-Indirect should be 1.0 for both
        '''
        config = self.get_default_config()
        # Test with G
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_a_Difference(self):
        config = self.get_default_config()
        # Switch both agents to D. Check rewards
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])

    def test_a_IndirectDifference(self):
        config = self.get_default_config()
        # Switch both agents to D-Indirect. Check rewards.
        # Should be 1.0 for rover and 1.0 for uav.
        # Rover gets credit for itself and uav gets credit for itself and the rover
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_a_IndirectDifference_manual_empty(self):
        """Manually assign no rovers to the uav"""
        config = self.get_default_config()
        # Switch both agents to D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        # Manually set up D-Indirect on the uav to get credit for no additional agents
        config['env']['agents']['uavs'][0]['IndirectDifference'] = {
            'type' : 'removal',
            'assignment' : 'manual',
            'manual' : [],
            'automatic' : {
                'timescale': '',
                'credit' : ''
            }
        }
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])

    def test_a_IndirectDifference_manual_0(self):
        """Manually assign the rover to the uav"""
        config = self.get_default_config()
        # Switch both agents to D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        # Manually set up D-Indirect on the uav to get credit for the rover
        config['env']['agents']['uavs'][0]['IndirectDifference'] = {
            'type' : 'removal',
            'assignment' : 'manual',
            'manual' : [0],
            'automatic' : {
                'timescale': '',
                'credit' : ''
            }
        }
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_a_Mixed(self):
        config = self.get_default_config()
        # Switch rover to D, uav to D-Indirect.
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_b(self):
        '''The rover is on the poi. The uav is far away
        '''
        config = self.get_default_config()
        # Move the uav far away
        config['env']['agents']['uavs'][0]['position']['fixed'] = [40.0, 40.0]
        # Check G
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
        # Switch both to D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])
        # Switch both to D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])
        # Switch just rover to D, leave uav as D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])

    def test_c(self):
        '''The uav is on the poi. The rover is far away
        '''
        config = self.get_default_config()
        # Move rover far away
        config['env']['agents']['rovers'][0]['position']['fixed'] = [50.0, 50.0]
        # Check G
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])
        # Switch both to D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])
        # Switch both to D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])
        # Switch just rover to D, leave uav using D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])

    def test_d(self):
        '''Neither the rover nor the uav are on the poi.
        They are too far to receive a reward
        '''
        config = self.get_default_config()
        # Move rover and uav far away
        config['env']['agents']['rovers'][0]['position']['fixed'] = [50.0, 50.0]
        config['env']['agents']['uavs'][0]['position']['fixed'] = [50.0, 50.0]
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])
        # Switch both to D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])
        # Switch both to D-Indirect
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[0.0, 0.0])

    def test_e(self):
        '''The rover is within the POI observation radius, but the uav is just outside it
        BUT, the rover is within the uav's influence radius
        '''
        config = self.get_default_config()
        # Move rover further from poi, still within poi observation radius
        config['env']['agents']['rovers'][0]['position']['fixed'] = [28.0, 25.0]
        # Move uav even farther, oustide poi observation radius
        config['env']['agents']['uavs'][0]['position']['fixed'] = [31.0, 25.0]
        # Check G
        G = self.compute_poi_reward_using_positions(
            config['env']['agents']['rovers'][0]['position']['fixed'],
            config['env']['pois']['hidden_pois'][0]['position']['fixed']
        )
        self.assert_correct_rewards(config, expected_rewards=[G, G])
        # Switch both to D and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[G, 0.0])
        # Switch both to D-Indirect and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[G, G])
        # Switch just rover back to D and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[G, G])

    def test_f(self):
        '''The rover is just barely within the POI observation radius, and just barely
        within the uav's influence radius
        '''
        config = self.get_default_config()
        # Move the rover further from the poi
        config['env']['agents']['rovers'][0]['position']['fixed'] = [30.0, 25.0]
        # Move the uav further from the rover
        config['env']['agents']['uavs'][0]['position']['fixed'] = [35.0, 25.0]
        # Check G
        G = self.compute_poi_reward_using_positions(
            config['env']['agents']['rovers'][0]['position']['fixed'],
            config['env']['agents']['uavs'][0]['position']['fixed']
        )
        self.assert_correct_rewards(config, expected_rewards=[G, G])
        # Switch both to D and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[G, 0.0])
        # Switch both to D-Indirect and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'IndirectDifference'
        config['env']['agents']['uavs'][0]['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[G, G])
        # Switch just rover back to D and check
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[G, G])

class TestTwoRoversTwoPois(TestFinalState):
    def get_config_a(self):
        config = self.get_env_template_config()
        # Place two hidden pois
        poi_config_0 = self.get_default_poi_config()
        poi_config_0['position']['fixed'] = [10.0, 10.0]
        config['env']['pois']['hidden_pois'].append(
            poi_config_0
        )
        poi_config_1 = deepcopy(poi_config_0)
        poi_config_1['position']['fixed'] = [10.0, 40.0]
        config['env']['pois']['hidden_pois'].append(
            poi_config_1
        )
        # Place two rovers. One at each poi
        rover_config_0 = self.get_default_rover_config()
        rover_config_0['position']['fixed'] = deepcopy(
            poi_config_0['position']['fixed']
        )
        config['env']['agents']['rovers'].append(
            rover_config_0
        )
        rover_config_1 = self.get_default_rover_config()
        rover_config_1['position']['fixed'] = deepcopy(
            poi_config_1['position']['fixed']
        )
        config['env']['agents']['rovers'].append(
            rover_config_1
        )
        # Set map bounds
        config['env']['map_size'] = [50.0, 50.0]
        return config

    def test_a_Global(self):
        config = self.get_config_a()
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0])

    def test_a_Difference(self):
        config = self.get_config_a()
        # Set rover rewards to Difference
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_a_IndirectDifference(self):
        config = self.get_config_a()
        # Set rover rewards to IndirectDifference
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'IndirectDifference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

if __name__ == '__main__':
    unittest.main()
