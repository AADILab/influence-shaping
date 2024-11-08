import unittest

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestSequentialDifference(TestEnv):
    """Run some simple checks to make sure D works properly when G is sequential
    - 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
    - 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
        D should update accordingly
    - 2 rovers, 2 POIs. Each rover visits its respective POI.
    - 2 rovers, 2 POIs. Each rover visits both POIs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'
        self.default_poi_config['position']['fixed'] = [10.0, 10.0]
        self.default_rover_config['position']['fixed'] = [40.0, 10.0]

    def get_default_config(self):
        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 50.]
        return config
    
    def get_one_rover_one_poi_config(self):
        config = self.get_default_config()
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config()]
        return config
    
    def get_two_rovers_one_poi_config(self):
        config = self.get_one_rover_one_poi_config()
        # Add a rover to config
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        return config
    
    def get_two_rovers_two_pois_config(self):
        config = self.get_two_rovers_one_poi_config()
        # Add a poi in a different location
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [10.0, 20.0]
        config['env']['pois']['rover_pois'].append(poi_config)
        return config

    def test_one_rover_one_poi(self):
        # -- 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
        config = self.get_one_rover_one_poi_config()
        # Run this with G
        env = createEnv(config)
        agent_paths = [
            [[20.0, 10.0]], # Rover moves closer to POI, but outside observation radius
            [[10.0, 10.0]], # Rover is on top of the POI
            [[40.0, 10.0]]  # Rover moves back to its starting position
        ]
        expected_rewards_at_each_step = [
            [0.0], # Initial setup. No reward.
            [0.0], # Rover is outside observation radius
            [1.0], # Rover observes POI
            [1.0]  # Rover moves out of observation radius, but POI is still counted
        ]
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
        # Run this with D. Same path. Same expected reward. (D should equal G with one agent)
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        env = createEnv(config)
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_two_rovers_one_poi(self):
        # -- 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
        config = self.get_two_rovers_one_poi_config()
        # Make the env and run it with G
        env = createEnv(config)
        agent_paths = [
            # Rover A moves on top of POI. Rover B stays still.
            [[10.0, 10.0], [40.0, 10.0]],
            # Rover A moves back to start. Rover B stays still.
            [[40.0, 10.0], [40.0, 10.0]],
            # Rover A stays still. Rover B moves on top of POI
            [[40.0, 10.0], [10.0, 10.0]],
            # Rover A stays still. Rover B returns to start
            [[40.0, 10.0], [40.0, 10.0]]
        ]
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover A moved onto POI. G is 1.0 for everyone from this point onwards
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
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
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_two_rovers_two_pois_a(self):
        # -- 2 rovers, 2 POIs. Each rover visits its respective POI.
        config = self.get_two_rovers_two_pois_config()
        # Make an env and run it with G
        env = createEnv(config)
        agent_paths = [
            # Rover A visits its POI. Rover B stays still
            [[10.0, 10.0], [40.0, 10.0]],
            # Rover A goes back home. Rover B stays still
            [[40.0, 10.0], [40.0, 10.0]],
            # Rover A stays home. Rover B visits the second POI
            [[40.0, 10.0], [10.0, 20.0]],
            # Rover A stays home. Rover B goes back home
            [[40.0, 10.0], [40.0, 10.0]]
        ]
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
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)
        # Switch rovers to D and run the env again
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
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step)

    def test_two_rovers_two_pois_b(self):
        # -- 2 rovers, 2 POIs. Each rover visits both POIs.
        config = self.get_two_rovers_two_pois_config()
        env = createEnv(config)
        agent_paths = [
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
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step,
            start_msg=f'Path rewards incorrect with 2 rovers and 2 POIs using G\n')
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
        self.assert_path_rewards(env, agent_paths, expected_rewards_at_each_step, 
            start_msg=f'Path rewards incorrect with 2 rovers and 2 POIs using D\n')

if __name__ == '__main__':
    unittest.main()