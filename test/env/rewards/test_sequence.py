import unittest
from copy import deepcopy

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestSequence(TestEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_one_rover_one_poi_config_a(self):
        # Set some defaults for the remainder of config
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'
        self.default_poi_config['position']['fixed'] = [10.0, 10.0]
        self.default_rover_config['position']['fixed'] = [40.0, 10.0]
    
        # Get env template and fill it
        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 50.]
        # Throw in a rover
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        # Throw in a poi
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config()]
        return config
    
    def get_one_rover_one_poi_config_b(self):
        # Change defaults for the remainder of config building
        self.default_rover_config['position']['fixed'] = [45.0, 25.0]
        self.default_poi_config['position']['fixed'] = [25.0, 25.0]
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'

        # Build out the config
        config = self.get_env_template_config()
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        config['env']['pois']['rover_pois'].append(self.get_default_poi_config())
        config['env']['map_size'] = [50., 50.]
        return config

    def get_one_rover_one_uav_five_pois_config_a(self):
        # Get env config started
        config = self.get_env_template_config()
        config['env']['map_size'] = [50.0, 50.0]

        # Set up rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [25.0, 25.0]
        rover_config['observation_radius'] = 5.0
        config['env']['agents']['rovers'].append(rover_config)

        # Set up uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = [23.0, 23.0]
        config['env']['agents']['uavs'].append(uav_config)

        # Set up pois (Approximately a pentagon)
        poi_positions = [
            [45. , 25.],
            [31. , 44.],
            [ 8. , 36.],
            [ 8. , 13.],
            [31. ,  5.]
        ]
        for position in poi_positions:
            poi_config = self.get_default_poi_config()
            poi_config['observation_radius'] = 5.0
            poi_config['capture_radius'] = 1000.0
            poi_config['position']['fixed'] = position
            config['env']['pois']['hidden_pois'].append(poi_config)
        
        return config

    def get_one_rover_two_pois_config_a(self):
        # Build on a prior config, and use those defaults
        config = self.get_one_rover_one_poi_config_b()
        new_poi_config = self.get_default_poi_config()
        new_poi_config['position']['fixed'] = [10.0, 25.0]
        config['env']['pois']['rover_pois'].append(
            new_poi_config
        )
        return config

    def get_two_rovers_one_poi_config_a(self):
        config = self.get_one_rover_one_poi_config_a()
        # Add a rover to config
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        return config
    
    def get_two_rovers_one_poi_config_b(self):
        config = self.get_one_rover_one_poi_config_b()
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        return config

    def get_two_rovers_two_pois_config_a(self):
        config = self.get_two_rovers_one_poi_config_a()
        # Add a poi in a different location
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [10.0, 20.0]
        config['env']['pois']['rover_pois'].append(poi_config)
        return config
    
    def get_two_rovers_two_uavs_six_pois_config_a(self):
        # Modify defaults for the remainder of the test where called
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'

        # Get the template config so we can fill it in
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

class TestOneRoverOnePoi(TestSequence):
    def get_config_a(self):
        return self.get_one_rover_one_poi_config_a()
    
    def get_config_b(self):
        return self.get_one_rover_one_poi_config_b()

    def get_path_a(self):
        # Path makes sense with config a
        return [
            [[20.0, 10.0]], # Rover moves closer to POI, but outside observation radius
            [[10.0, 10.0]], # Rover is on top of the POI
            [[40.0, 10.0]]  # Rover moves back to its starting position
        ]

    def get_path_b(self):
        # Path makes sense with config b
        return [
            [[28.0, 25.0]], # Rover moves towards poi within the observation radius
            [[45.0, 25.0]], # Rover moves away outside the observation radius. Reward should not change
            [[25.0, 25.0]], # Rover moves directly on top of the poi. Reward should be 1.0
            [[45.0, 25.0]] # Rover moves away from poi again. Reward should remain at 1.0
        ]

    def test_config_a_path_a_Global(self):
        # -- 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
        config = self.get_config_a()
        env = createEnv(config)
        agent_paths = self.get_path_a()
        expected_rewards_at_each_step = [
            [0.0], # Initial setup. No reward.
            [0.0], # Rover is outside observation radius
            [1.0], # Rover observes POI
            [1.0]  # Rover moves out of observation radius, but POI is still counted
        ]
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
    
    def test_config_a_path_a_Difference(self):
        # -- 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
        config = self.get_config_a()
        # Run this with D. Same path. Same expected reward. (D should equal G with one agent)
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        expected_rewards_at_each_step = [
            [0.0], # Initial setup. No reward.
            [0.0], # Rover is outside observation radius
            [1.0], # Rover observes POI
            [1.0]  # Rover moves out of observation radius, but POI is still counted
        ]
        env = createEnv(config)
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)

    def test_config_b_path_b_Global(self):
        # Rover goes back and forth to and from a poi
        config = self.get_config_b()
        env = createEnv(config)
        partial_reward = self.compute_poi_reward_using_positions([28.0, 25.0],[25.0, 25.0])
        expected_rewards_at_each_step = [
            [0.0], # Rover starts far from poi. No reward
            [partial_reward], # Rover moves towards poi within the observation radius
            [partial_reward], # Rover moves away outside the observation radius. Reward should not change
            [1.0], # Rover moves directly on top of the poi. Reward should be 1.0
            [1.0] # Rover moves away from poi again. Reward should remain at 1.0
        ]   
        self.assert_path_rewards(env, self.get_path_b(), expected_rewards_at_each_step)

class TestOneRoverOneUavFivePois(TestSequence):
    def get_config_a(self):
        return self.get_one_rover_one_uav_five_pois_config_a()
    
    def get_path_a(self):
        # Rover and uav seperate and rover captures pois by itself
        return [
            # Rover captures first poi. Uav goes to corner of the map
            [[45. , 25.], [0.0, 0.0]],
            # Rover captures second poi
            [[31. , 44.], [0.0, 0.0]],
            # third
            [[ 8. , 36.], [0.0, 0.0]],
            # fourth
            [[ 8. , 13.], [0.0, 0.0]],
            # fifth. Uav has not moved since it went to the corner
            [[31. ,  5.], [0.0, 0.0]]
        ]

    def test_config_a_path_a_Global(self):
        # Make sure global reward is computed properly here
        config = self.get_config_a()
        env = createEnv(config)
        agent_paths = self.get_path_a()

        # Build out the expected rewards
        # Reward is just based on distance from rover to each poi
        expected_rewards_at_each_step = []
        start_positions = [[
            config['env']['agents']['rovers'][0]['position']['fixed'],
            config['env']['agents']['uavs'][0]['position']['fixed']
        ]]
        for agent_positions in start_positions+agent_paths:
            rover_position = agent_positions[0]
            G_at_this_step = 0.0
            for poi_config in config['env']['pois']['hidden_pois']:
                poi_position = poi_config['position']['fixed']
                G_at_this_step += self.compute_poi_reward_using_positions(rover_position, poi_position)
            # Each agent gets the G at this step
            expected_rewards_at_each_step.append([G_at_this_step, G_at_this_step])
        
        # Now check it
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_config_a_path_a_IndirectDifference(self):
        # This should be the same as G. 
        # Rover and uav start next to each other, so uav should get credit for the rover
        # using the default D-Indirect reward
        config = self.get_config_a()

        # Make sure to switch agents to D-Indirect!
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'

        env = createEnv(config)
        agent_paths = self.get_path_a()

        # Build out the expected rewards
        # Reward is just based on distance from rover to each poi
        expected_rewards_at_each_step = []
        start_positions = [[
            config['env']['agents']['rovers'][0]['position']['fixed'],
            config['env']['agents']['uavs'][0]['position']['fixed']
        ]]
        for agent_positions in start_positions+agent_paths:
            rover_position = agent_positions[0]
            G_at_this_step = 0.0
            for poi_config in config['env']['pois']['hidden_pois']:
                poi_position = poi_config['position']['fixed']
                G_at_this_step += self.compute_poi_reward_using_positions(rover_position, poi_position)
            # Each agent gets the G at this step
            expected_rewards_at_each_step.append([G_at_this_step, G_at_this_step])
        
        # Now check it
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

class TestOneRoverTwoPois(TestSequence):
    def get_config_a(self):
        return self.get_one_rover_two_pois_config_a()
    
    def get_path_a(self):
        return [
            [[28.0, 25.0]], # Rover moves nearby first poi
            [[25.0, 25.0]], # Rover moves on top of first poi
            [[13.5, 25.0]], # Rover moves closer to second poi.
            [[10.0, 25.0]], # Rover moves on top of second poi
            [[45.0, 25.0]] # Rover moves back to its starting point
        ]

    def test_config_a_Global(self):
        config = self.get_config_a()
        env = createEnv(config)
        partial_reward_first_poi = self.compute_poi_reward_using_positions([28.0, 25.0], [25.0, 25.0])
        partial_reward_second_poi = 1. + self.compute_poi_reward_using_positions([13.5, 25.0],[10.0, 25.0])
        expected_rewards_at_each_step = [
            [0.0], # Rover starts far from poi. No reward
            [partial_reward_first_poi], # Rover moves nearby first poi. Rewarded for that poi
            [1.0], # Rover moves on top of first poi. Reward is 1.0
            [partial_reward_second_poi], # Rover moves closer to second poi. Reward of 1.0 for first poi, and partial reward for second poi
            [2.0], # Rover moves on top of second poi. Reward of 2.0, 1.0 for each poi
            [2.0] # Rover moves back to its starting poin. Reward should remain the same
        ]
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)

class TestTwoRoversOnePoi(TestSequence):
    def get_config_a(self):
        return self.get_two_rovers_one_poi_config_a()
    
    def get_config_b(self):
        return self.get_two_rovers_one_poi_config_b()
    
    def get_path_a(self):
        # Used with config a
        return  [
            # Rover A moves on top of POI. Rover B stays still.
            [[10.0, 10.0], [40.0, 10.0]],
            # Rover A moves back to start. Rover B stays still.
            [[40.0, 10.0], [40.0, 10.0]],
            # Rover A stays still. Rover B moves on top of POI
            [[40.0, 10.0], [10.0, 10.0]],
            # Rover A stays still. Rover B returns to start
            [[40.0, 10.0], [40.0, 10.0]]
        ]
    
    def get_path_b(self):
        # Used with config b
        return [
            # Rover A gets within the poi's observation radius. Rover B stays still at start location
            [[28.0, 25.0], [45.0, 25.0]],
            # Rover A returns home. Rover B stays still
            [[45.0, 25.0], [45.0, 25.0]],
            # Rover A stays. Rover B gets close to POI than rover A got
            [[45.0, 25.0], [27.0, 25.0]],
            # Rover A stays still. Rover B leaves the POI
            [[45.0, 25.0], [45.0, 25.0]]
        ]
    
    def test_config_a_path_a_Global(self):
        # -- 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
        config = self.get_config_a()
        # Make the env and run it with G
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover A moved onto POI. G is 1.0 for everyone from this point onwards
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)

    def test_config_a_path_a_Difference(self):
        # -- 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
        config = self.get_config_a()
        # Switch both rovers to D
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        # Make the env and run it with D. Same agent paths
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover A moved onto POI. D is 1.0 for Rover A now
            [1.0, 0.0],
            # Rover A moves off POI. D is still 1.0 for Rover A
            [1.0, 0.0],
            # Rover B moves onto POI. D is 0.0 for everyone from this point onwards
            # (Nobody gets credit for poi observation. You remove either agent and POI is still observed)
            [0.0, 0.0],
            [0.0, 0.0]
        ]
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)

    def test_config_b_path_b_Global(self):
        # -- Rover A goes to a poi, then leaves. Rover B gets closer to that poi, then leaves.
        config = self.get_config_b()
        env = createEnv(config)
        partial_reward_1 = self.compute_poi_reward_using_positions([28.0, 25.0], [25.0, 25.0])
        partial_reward_2 = self.compute_poi_reward_using_positions([27.0, 25.0], [25.0, 25.0])
        expected_rewards_at_each_step = [
            # Initial setup. Neither rover has observed a poi. No reward at the start
            [0.0, 0.0],
            # Rover A gets within the poi's observation radius. Rewards go up
            [partial_reward_1, partial_reward_1],
            # Rover A leaves. Rover B stays still. Rewards remain the same
            [partial_reward_1, partial_reward_1],
            # Rover A stays still. Rover B gets closer than rover A got. Rewards go up
            [partial_reward_2, partial_reward_2],
            # Rover A stays still. Rover B leaves the POI. Rewards stay the same
            [partial_reward_2, partial_reward_2]
        ]
        self.assert_path_rewards(env, self.get_path_b(), expected_rewards_at_each_step)
    
class TestTwoRoversTwoPois(TestSequence):
    def get_config_a(self):
        return self.get_two_rovers_two_pois_config_a()

    def get_path_a(self):
        # -- 2 rovers, 2 POIs. Each rover visits its respective POI.
        return [
            # Rover A visits its POI. Rover B stays still
            [[10.0, 10.0], [40.0, 10.0]],
            # Rover A goes back home. Rover B stays still
            [[40.0, 10.0], [40.0, 10.0]],
            # Rover A stays home. Rover B visits the second POI
            [[40.0, 10.0], [10.0, 20.0]],
            # Rover A stays home. Rover B goes back home
            [[40.0, 10.0], [40.0, 10.0]]
        ]

    def get_path_b(self):
        # -- 2 rovers, 2 POIs. Each rover visits both POIs.
        return [
            # Rover A visits the first POI. Rover B stays home
            [[10.0, 10.0], [40.0, 10.0]],
            # Rover A visits the second POI. Rover B stays home
            [[10.0, 20.0], [40.0, 10.0]],
            # Rover A goes home. Rover B stays home
            [[40.0, 10.0], [40.0, 10.0]],
            # Rover A stays home. Rover B visits first POI
            [[40.0, 10.0], [10.0, 10.0]],
            # Rover A stays home. Rover B visits second POI
            [[40.0, 10.0], [10.0, 20.0]],
            # Rover A stays home. Rover B goes back home
            [[40.0, 10.0], [40.0, 10.0]]
        ]
    
    def test_config_a_path_a_Global(self):
        # -- 2 rovers, 2 POIs. Each rover visits its respective POI.
        config = self.get_config_a()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. Rovers are far from POIs. No reward.
            [0.0, 0.0],
            # Rover A visited POI. G goes up
            [1.0, 1.0],
            # Rover A goes home. G remains the same.
            [1.0, 1.0],
            # Rover B visits the second POI. G goes up
            [2.0, 2.0],
            # Rover B goes home. G remains the same
            [2.0, 2.0]
        ]
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)
    
    def test_config_a_path_a_Difference(self):
        # -- 2 rovers, 2 POIs. Each rover visits its respective POI.
        config = self.get_config_a()
        # Switch rovers to D and run the env
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial Setup. Too far to get rewards.
            [0.0, 0.0],
            # Rover A visited POI. D for Rover A goes up
            [1.0, 0.0],
            # Rover A goes home. No changes in D
            [1.0, 0.0],
            # Rover B visits second POI. D for Rover B goes up
            [1.0, 1.0],
            # Rover B goes home. No changes in D
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a(), expected_rewards_at_each_step)

    def test_config_a_path_b_Global(self):
        # -- 2 rovers, 2 POIs. Each rover visits both POIs.
        config = self.get_config_a()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. Too far to get rewards
            [0.0, 0.0],
            # Rover A visits the first POI. G is 1.0
            [1.0, 1.0],
            # Rover A visits second POI. G stays the same here on out
            [2.0, 2.0],
            # Rover A goes home.
            [2.0, 2.0],
            # Rover B visits first POI.
            [2.0, 2.0],
            # Rover B visits second POI
            [2.0, 2.0],
            # Rover B goes home
            [2.0, 2.0]
        ]
        self.assert_path_rewards(env, self.get_path_b(), expected_rewards_at_each_step,
            start_msg=f'Path rewards incorrect with 2 rovers and 2 POIs using G\n')

    def test_config_a_path_b_Difference(self):
        # -- 2 rovers, 2 POIs. Each rover visits both POIs.
        config = self.get_config_a()
        # Now check with D
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. Too far to get rewards
            [0.0, 0.0],
            # Rover A visits first POI, gets D of 1.0
            [1.0, 0.0],
            # Rover A visits second POI. Gets D of 2.0
            [2.0, 0.0],
            # Rover A goes home. D stays the same
            [2.0, 0.0],
            # Rover B visits first POI. No D for rover B
            # D for Rover A goes down because first POI is observed by rover B as well
            [1.0, 0.0],
            # D goes down for rover A because second POI is now also observed by rover B
            [0.0, 0.0],
            # Rover B goes home. D is 0.0 for both rovers
            # If you remove either rover, the POIs are still observed by the other rover.
            [0.0, 0.0]
        ]
        self.assert_path_rewards(env, self.get_path_b(), expected_rewards_at_each_step, 
            start_msg=f'Path rewards incorrect with 2 rovers and 2 POIs using D\n')

class TestTwoRoversTwoUavsSixPois(TestSequence):
    """Check that rewards are computed as expected for sequentially observable POIs
    2 rovers, 2 uavs, 6 pois
    """
    def get_config_a(self):
        return self.get_two_rovers_two_uavs_six_pois_config_a()
    
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

    def test_config_a_path_a_Global(self):
        """Keep agents using G and compute rewards"""
        config = self.get_config_a()
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
    
    def test_config_a_path_a_Global_final(self):
        """Make sure the final G rewards match when we don't compute rewards at each step"""
        config = self.get_config_a()
        self.assert_final_rewards(createEnv(config), self.get_path_a(), expected_final_rewards = [6.0, 6.0, 6.0, 6.0])

    def test_config_a_path_a_Difference(self):
        """Switch agents to D"""
        config = self.get_config_a()
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
    
    def test_config_a_path_a_Difference_final(self):
        """Make sure the final D rewards match when we don't compute rewards at each step"""
        config = self.get_config_a()
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'Difference'
        self.assert_final_rewards(createEnv(config), self.get_path_a(), expected_final_rewards=[3.0, 3.0, 0.0, 0.0])
    
    def test_config_a_path_a_Mixed(self):
        """Switch rovers to D. Keep uavs using G"""
        config = self.get_config_a()
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
    
    def test_config_a_path_a_Mixed_final(self):
        """Make sure final mixed G,D rewards match when we don't compute rewards each step"""
        config = self.get_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_final_rewards(createEnv(config), self.get_path_a(), expected_final_rewards=[3.0, 3.0, 6.0, 6.0])
    
    def test_config_a_path_a_IndirectDifferenceAutomatic(self):
        """Using Default D-Indirect. Trajectory based, all or nothing credit, remove agents you get credit for"""
        config = self.get_config_a()
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
    
    def test_config_a_path_a_IndirectDifferenceAutomatic_final(self):
        config = self.get_config_a()
        for agent_config in config['env']['agents']['rovers']+config['env']['agents']['uavs']:
            agent_config['reward_type'] = 'IndirectDifference'
        self.assert_final_rewards(createEnv(config), self.get_path_a(), expected_final_rewards=[3.0, 3.0, 3.0, 3.0])

    def test_config_a_path_a_IndirectDifferenceAutomaticTimestep(self):
        """Using D-Indirect that computes influence based on individual timesteps"""
        config = self.get_config_a()
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
    
    def test_config_a_path_a_IndirectDifferenceAutomaticTimestep(self):
        config = self.get_config_a()
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
        self.assert_final_rewards(createEnv(config), self.get_path_a(), expected_final_rewards=[3.0, 3.0, 3.0, 3.0])

    def test_config_a_path_a_IndirectDifferenceManual_0to0_1to1(self):
        """Using D-Indirect with manual assignment of rovers to uavs
        Uav 0 gets credit for rover 0
        Uav 1 gets credit for rover 1

        IE: Each uav gets credit for one rover
        """
        config = self.get_config_a()
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

    def test_config_a_path_a_IndirectDifferenceManual_1to0_0to1(self):
        """Using D-Indirect with manual assignment of rovers to uavs
        Uav 1 gets credit for rover 0
        Uav 0 gets credit for rover 1

        IE: Each uav gets credit for one rover (but flipped)
        """
        config = self.get_config_a()
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

    def test_config_a_path_a_IndirectDifferenceManual_0to0and1(self):
        """Uav 0 gets credit for rovers 0 and 1"""
        config = self.get_config_a()
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

    def test_config_a_path_a_IndirectDifferenceManual_1to0and1(self):
        """Uav 1 gets credit for rovers 0 and 1"""
        config = self.get_config_a()
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
            [[90.0, 90.0], [60.0, 50.0], [10.0, 90.0], [90.0, 90.0]],
            # Rover A visits middle left POI. Joined by uav A
            [[10.0, 50.0], [60.0, 50.0], [10.0, 50.0], [90.0, 90.0]],
            # Rover A visits middle right POI. Joined by uav B
            [[90.0, 50.0], [60.0, 50.0], [10.0, 50.0], [90.0, 50.0]],
            # Rover A visits bottom left POI. Joined by uav A
            [[10.0, 10.0], [60.0, 50.0], [10.0, 10.0], [90.0, 50.0]],
            # Rover A visits bottom right POI. Joined by uav B
            [[90.0, 10.0], [60.0, 50.0], [10.0, 10.0], [90.0, 10.0]]
        ]
    
    def test_config_a_path_b_Global(self):
        """Global rewards"""
        config = self.get_config_a()
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

    def test_config_a_path_b_IndirectDifferenceAutomatic(self):
        """Default trajectory based D-Indirect"""
        config = self.get_config_a()
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
    
    # @unittest.skip("This feature isn't ready yet, so no need to test (yet).")
    def test_config_a_path_b_IndirectDifferenceAutomaticTimestep(self):
        """Now with granular credit assignment. Each uav gets credit for the pois it 'helped' with"""
        config = self.get_config_a()
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
