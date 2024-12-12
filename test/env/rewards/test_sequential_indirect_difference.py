import unittest
from copy import deepcopy

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestSequentialIndirectDifference(TestEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'

class TestTwoRoversTwoUavsSixPois(TestSequentialIndirectDifference):
    """Check that D-Indirect is computed as expected for sequentially observable POIs
    2 rovers, 2 uavs, 6 pois
    """
    def get_default_config(self):
        config = self.get_env_template_config()
        # Add the rovers
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        config['env']['agents']['rovers'][0]['position']['fixed'] = [50.0, 50.0]
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        config['env']['agents']['rovers'][1]['position']['fixed'] = [60.0, 50.0]
        # Add the uavs
        config['env']['agents']['uavs'].append(self.get_default_uav_config())
        config['env']['agents']['uavs'][0]['position']['fixed'] = [48.0, 48.0]
        config['env']['agents']['uavs'].append(self.get_default_uav_config())
        config['env']['agents']['uavs'][1]['position']['fixed'] = [58.0, 48.0]
        # Add the pois
        poi_positions = [
            [10.0, 10.0],
            [10.0, 50.0],
            [10.0, 90.0],
            [90.0, 10.0],
            [90.0, 50.0],
            [90.0, 90.0]
        ]
        for ind, poi_position in enumerate(poi_positions):
            config['env']['pois']['hidden_pois'].append(self.get_default_poi_config())
            config['env']['pois']['hidden_pois'][ind]['position']['fixed'] = poi_position
        config['env']['map_size'] = [100.0, 100.0]
        return config
    
    def get_path_a(self):
        """This is an ideal path where each rover-uav pair observe the set of POIs on their side"""
        return [
            # Rover A and uav A visit top POI on the left.
            # Rover B and uav B visit top POI on the right
            [[10.0, 90.0], [90.0, 90.0], [10.0, 90.0], [90.0, 90.0]],
            # Rover-uav pairs continue to middle POIs
            [[10.0, 50.0], [90.0, 50.0], [10.0, 50.0], [90.0, 50.0]],
            # Rover-uav pairs continue to final POIs on the bottom
            [[10.0, 10.0], [90.0, 10.0], [10.0, 10.0], [90.0, 10.0]]
        ]

    def test_a_Global(self):
        """Keep agents using G and compute rewards"""
        config = self.get_default_config()
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. Too far to get rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. G is 2.0
            [2.0, 2.0, 2.0, 2.0],
            # Second two pois visited. G is 4.0 (up by 2.0)
            [4.0, 4.0, 4.0, 4.0],
            # Last two pois visited. G is 6.0 (up by 2.0 more)
            [6.0, 6.0, 6.0, 6.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_Difference(self):
        """Switch agents to D"""
        config = self.get_default_config()
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'Difference'
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. Too far to get rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Rovers get credit through D
            [1.0, 1.0, 0.0, 0.0],
            # Second two pois visited. Rovers each get credit for an additional POI
            [2.0, 2.0, 0.0, 0.0],
            # Last two pois visited. Rovers each get credit for 3 pois
            [3.0, 3.0, 0.0, 0.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
    
    def test_a_Mixed(self):
        """Switch rovers to D. Keep uavs using G"""
        config = self.get_default_config()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Each rover gets credit. Uavs get G of 2.0
            [1.0, 1.0, 2.0, 2.0],
            # Middle two pois visited. Each rover gets credit for 2 pois. Uavs get G of 4.0
            [2.0, 2.0, 4.0, 4.0],
            # Last two pois visited
            [3.0, 3.0, 6.0, 6.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
    
    def test_a_IndirectDifferenceAutomatic(self):
        """Using Default D-Indirect. Trajectory based, all or nothing credit, remove agents you get credit for"""
        config = self.get_default_config()
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Each rover-uav pair gets credit for 1 poi
            [1.0, 1.0, 1.0, 1.0],
            # Middle two pois visited. Each rover-uav pair gets credit for 2 pois
            [2.0, 2.0, 2.0, 2.0],
            # Last two pois visited. Each rover-uav pair gets credit for 3 pois
            [3.0, 3.0, 3.0, 3.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_IndirectDifferenceAutomaticTimestep(self):
        """Using D-Indirect that computes influence based on individual timesteps"""
        config = self.get_default_config()
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [0],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Each rover-uav pair gets credit for 1 poi
            [1.0, 1.0, 1.0, 1.0],
            # Middle two pois, each rover-uav pair gets credit for 2 pois
            [2.0, 2.0, 2.0, 2.0],
            # Last two pois
            [3.0, 3.0, 3.0, 3.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_IndirectDifferenceManual_0to0_1to1(self):
        """Using D-Indirect with manual assignment of rovers to uavs
        Uav 0 gets credit for rover 0
        Uav 1 gets credit for rover 1

        IE: Each uav gets credit for one rover
        """
        config = self.get_default_config()
        # Switch everyone to D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        # Set D-Indirect to manual for the uavs
        for uav_config in config['env']['agents']['uavs']:
            uav_config['IndirectDifference']['assignment'] = 'manual'
        # Assign rovers to uavs
        config['env']['agents']['uavs'][0]['IndirectDifference']['manual'] = [0]
        config['env']['agents']['uavs'][1]['IndirectDifference']['manual'] = [1]
        # Set up paths and rewards. Then test
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Each rover-uav pair gets credit for 1 poi
            [1.0, 1.0, 1.0, 1.0],
            # Middle two pois, each rover-uav pair gets credit for 2 pois
            [2.0, 2.0, 2.0, 2.0],
            # Last two pois
            [3.0, 3.0, 3.0, 3.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_IndirectDifferenceManual_1to0_0to1(self):
        """Using D-Indirect with manual assignment of rovers to uavs
        Uav 1 gets credit for rover 0
        Uav 0 gets credit for rover 1

        IE: Each uav gets credit for one rover (but flipped)
        """
        config = self.get_default_config()
        # Switch everyone to D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        # Set D-Indirect to manual for uavs
        for uav_config in config['env']['agents']['uavs']:
            uav_config['IndirectDifference']['assignment'] = 'manual'
        # Assign rovers to uavs
        config['env']['agents']['uavs'][1]['IndirectDifference']['manual'] = [0]
        config['env']['agents']['uavs'][0]['IndirectDifference']['manual'] = [1]
        # Set up paths and rewards, then test
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. 
            # Still a 1:1 mapping of rovers to uavs, 
            # just that each uav gets credit for the OTHER rover now
            [1.0, 1.0, 1.0, 1.0],
            # Middle pois
            [2.0, 2.0, 2.0, 2.0],
            # Final pois
            [3.0, 3.0, 3.0, 3.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_IndirectDifferenceManual_0to0and1(self):
        """Uav 0 gets credit for rovers 0 and 1"""
        config = self.get_default_config()
        # Switch everyone to D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        # Set D-Indirect to manual for uavs
        for uav_config in config['env']['agents']['uavs']:
            uav_config['IndirectDifference']['assignment'] = 'manual'
        # Assign rovers to uavs
        config['env']['agents']['uavs'][0]['IndirectDifference']['manual'] = [0,1]
        # Set up paths and rewards. Then test
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Uav 0 gets credit for both rovers. Uav 1 gets no credit
            [1.0, 1.0, 2.0, 0.0],
            # Middle pois
            [2.0, 2.0, 4.0, 0.0],
            # Final pois
            [3.0, 3.0, 6.0, 0.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_a_IndirectDifferenceManual_1to0and1(self):
        """Uav 1 gets credit for rovers 0 and 1"""
        config = self.get_default_config()
        # Switch everyone to D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        # Set D-Indirect to manual for uavs
        for uav_config in config['env']['agents']['uavs']:
            uav_config['IndirectDifference']['assignment'] = 'manual'
        # Assign rovers to uavs
        config['env']['agents']['uavs'][1]['IndirectDifference']['manual'] = [0,1]
        # Set up paths and rewards. Then test
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # First two pois visited. Uav 1 gets credit for both rovers. Uav 0 gets no credit
            [1.0, 1.0, 0.0, 2.0],
            # Middle pois
            [2.0, 2.0, 0.0, 4.0],
            # Final pois
            [3.0, 3.0, 0.0, 6.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
    
    def get_path_b(self):
        """Rover A observes all of the POIs with support from both uavs at different points in time"""
        return [
            # Rover A and uav A visit top POI on the left.
            # Rover B and uav B stay still
            [[10.0, 90.0], [60.0, 50.0], [10.0, 90.0], [58.0, 48.0]],
            # Rover A visits top POI on the right, joined by uav B
            [[90.0, 90.0], [60.0, 50.0], [10.0, 90.0], [90.0, 10.0]],
            # Rover A visits middle left POI. Joined by uav A
            [[10.0, 50.0], [60.0, 50.0], [10.0, 50.0], [90.0, 10.0]],
            # Rover A visits middle right POI. Joined by uav B
            [[90.0, 50.0], [60.0, 50.0], [10.0, 50.0], [90.0, 50.0]],
            # Rover A visits bottom left POI. Joined by uav A
            [[10.0, 10.0], [60.0, 50.0], [10.0, 10.0], [90.0, 50.0]],
            # Rover A visits bottom right POI. Joined by uav B
            [[90.0, 10.0], [60.0, 50.0], [10.0, 10.0], [90.0, 50.0]]
        ]
    
    def test_b_Global(self):
        """Global rewards"""
        config = self.get_default_config()
        agent_paths = self.get_path_b()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # Rover A and uav A visit top left POI. One POI observed
            [1.0, 1.0, 1.0, 1.0],
            # Rover A and uav B visit top right POI. Two pois observed
            [2.0, 2.0, 2.0, 2.0],
            # Rover A and uav A visit middle left POI. Three pois observed
            [3.0, 3.0, 3.0, 3.0],
            # Rover A and uav B visit middle right POI. Four pois observed
            [4.0, 4.0, 4.0, 4.0],
            # Rover A and uav A visit bottom left POI. Five pois observed
            [5.0, 5.0, 5.0, 5.0],
            # Rover A and uav B visit bottom right POI. Six pois observed
            [6.0, 6.0, 6.0, 6.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_b_IndirectDifferenceAutomatic(self):
        """Default trajectory based D-Indirect"""
        config = self.get_default_config()
        # Switch all agents to default D-Indirect
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        agent_paths = self.get_path_b()
        expected_rewards_at_each_step = [
            # Initial setup. No rewards
            [0.0, 0.0, 0.0, 0.0],
            # Uav A gets all the credit for rover A based on counters
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [3.0, 0.0, 3.0, 0.0],
            [4.0, 0.0, 4.0, 0.0],
            [5.0, 0.0, 5.0, 0.0],
            [6.0, 0.0, 6.0, 0.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
    
    def test_b_IndirectDifferenceAutomaticTimestep(self):
        """Now with granular credit assignment. Each uav gets credit for the pois it 'helped' with"""
        config = self.get_default_config()
        # Switch all agents to D-Indirect (with timestep based/ more granular credit assignment)
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
            agent_config['IndirectDifference'] = {
                'type': 'removal',
                'assignment': 'automatic',
                'manual': [],
                'automatic': {
                    'timescale': 'timestep',
                    'credit': 'AllOrNothing'
                }
            }
        agent_paths = self.get_path_b()
        expected_rewards_at_each_step = [
            # Initial setup, no rewards
            [0.0, 0.0, 0.0, 0.0],
            # Uav A gets credit for rover A's observation
            [1.0, 0.0, 1.0, 0.0],
            # Uav A gets credit for first observation, uav B gets credit for second observation
            [2.0, 0.0, 1.0, 1.0],
            # Uav A gets credit for first and third observations, uav B gets credit for second
            [3.0, 0.0, 2.0, 1.0],
            # Uav A gets credit for first, third, uav B gets credit for second, fourth
            [4.0, 0.0, 2.0, 2.0],
            # Uav A gets credit for first, third, fifth, uav B gets credit for second, fourth
            [5.0, 0.0, 3.0, 2.0],
            # Uav A gets credit for first, third, fifth, uav B gets credit for second, fourth, sixth
            [6.0, 0.0, 3.0, 3.0]
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

if __name__ == '__main__':
    unittest.main()
