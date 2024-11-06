import unittest
import numpy as np
from copy import deepcopy
from influence.librovers import rovers
from influence.custom_env import createEnv
from influence.testing import TestEnv

class TestRewards(TestEnv):
    def test_G(self):
        """Check that G is computed as expected for different cases
        - additional rovers with low coupling (if coupling is satisfied, addtl rovers shouldn't change anything)
        - poi observation radii
        - uavs in the environment (should not affect G)
        - more pois
        """
        self.default_uav_config['position']['fixed'] = [23.0, 24.0]
        def get_default_config(self):
            config = self.get_env_template_config()
            config['env']['agents']['rovers'].append(self.get_default_rover_config())
            config['env']['agents']['uavs'] = []
            config['env']['pois']['rover_pois'].append(self.get_default_poi_config())
            config['env']['map_size'] = [50., 50.]
            return config

        # Check one rover, one poi. Reward of 1.0
        config = get_default_config(self)
        self.assert_correct_rewards(config, expected_rewards=[1.0])

        # Place the rover closer to the poi. Reward should still be 1.0
        config['env']['agents']['rovers'][0]['position']['fixed'] = [24.5, 24.0]
        self.assert_correct_rewards(config, expected_rewards=[1.0])

        # Place the rover further away. Reward should be 0.5
        config['env']['agents']['rovers'][0]['position']['fixed'] = [26.0, 24.0]
        self.assert_correct_rewards(config, expected_rewards=[0.5])

        # Make the rover follow a path. 
        # Reward should be 0.5 based on final position in path to poi
        rover_path = [
            [24.5, 24.0],
            [26.0, 24.0]
        ]
        config = get_default_config(self)
        env = createEnv(config)
        _, _ = env.reset()
        for position in rover_path:
            env.rovers()[0].set_position(position[0], position[1])
        reward = self.compute_G(env)
        expected_reward = 0.5
        self.assert_np_close(reward, expected_reward)

        # Now let's add another copy of our rover to make sure we are not double counting
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

        # Add another 10 and check again for good measure
        for _ in range(10):
            config['env']['agents']['rovers'].append(self.get_default_rover_config())
        self.assert_correct_rewards(config, expected_rewards=[1.0]*12)

        # Reset the config, and bring down the observation radius for the poi
        # G should still be 1.0
        config = get_default_config(self)
        config['env']['pois']['rover_pois'][0]['observation_radius'] = 5.0
        env = createEnv(config)
        _, (reward) = env.reset()
        expected_reward = 1.0
        self.assert_np_close(reward, expected_reward)
        
        # But if the rover moves away, G should go down
        env.rovers()[0].set_position(27.0, 24.0)
        reward = self.compute_G(env)
        expected_reward = self.compute_poi_reward(env.pois()[0], env.rovers()[0])
        self.assert_np_close(reward, expected_reward)

        # If the rover moves far enough away, G becomes 0.0
        env.rovers()[0].set_position(30.0, 24.0)
        reward = self.compute_G(env)
        expected_reward = 0.0
        self.assert_np_close(reward, expected_reward)

        # Let's set the default config and throw in some uavs. Should not change G
        config = get_default_config(self)
        expected_reward = 1.0
        for _ in range(3):
            config['env']['agents']['uavs'].append(self.get_default_uav_config())
            env = createEnv(config)
            _, rewards = env.reset()
            reward = rewards[0]
            self.assert_np_close(reward, expected_reward)
        
        # Let's set the default config and add more pois.
        config = get_default_config(self)
        more_poi_positions = [
            [25.0, 35.0],
            [35.0, 35.0]
        ]
        poi_config = self.get_default_poi_config()
        for position in more_poi_positions:
            poi_config['position']['fixed'] = position
            config['env']['pois']['rover_pois'].append(poi_config)
        # With only 1 rover, G should be 1.0 + the value of the other two pois
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 1.0 + \
            self.compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            self.compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assert_np_close(reward, expected_reward)

        # Let's add uavs to observe the remaining pois 
        # (reward should not change because uavs can't observe pois)
        uav_config = self.get_default_uav_config()
        for position in more_poi_positions:
            uav_config['position']['fixed'] = position
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 1.0 + \
            self.compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            self.compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assert_np_close(reward, expected_reward)

        # Now let's add rovers to actually observe the remaining pois
        rover_config = self.get_default_rover_config()
        for position in more_poi_positions:
            rover_config['position']['fixed'] = position
            config['env']['agents']['rovers'].append(rover_config)
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 3.0
        self.assert_np_close(reward, expected_reward)

    def test_sequential_G(self):
        """Check that G is computed as expected when it is sequential
        - rover goes back and forth to and from a poi
        - rover goes to poi A, then poi B, then returns to starting point
        - rover A goes to poi, then leaves. rover B gets closer to poi, then leaves
        """
        self.default_rover_config['position']['fixed'] = [45.0, 25.0]
        self.default_poi_config['position']['fixed'] = [25.0, 25.0]
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'
        def get_default_config(self):
            config = self.get_env_template_config()
            config['env']['agents']['rovers'].append(self.get_default_rover_config())
            config['env']['pois']['rover_pois'].append(self.get_default_poi_config())
            config['env']['map_size'] = [50., 50.]
            return config
        # -- Rover goes back and forth to and from a poi
        config = get_default_config(self)
        env = createEnv(config)
        _, _ = env.reset()
        # Rover starts far from poi. No reward
        reward = self.compute_G(env)
        expected_reward = 0.0
        self.assert_np_close(reward, expected_reward)
        # Rover moves towards poi within the observation radius
        env.rovers()[0].set_position(28.0, 25.0)
        reward = self.compute_G(env)
        rover_position = [env.rovers()[0].position().x, env.rovers()[0].position().y]
        poi_position = [env.pois()[0].position().x, env.pois()[0].position().y]
        expected_reward = self.compute_poi_reward_using_positions(rover_position, poi_position)
        self.assert_np_close(reward, expected_reward)
        # Rover moves away outside the observation radius. Reward should not change
        env.rovers()[0].set_position(45.0, 25.0)
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        # Rover moves directly on top of the poi. Reward should be 1.0
        env.rovers()[0].set_position(25.0, 25.0)
        expected_reward = 1.0
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        # Rover moves away from poi again. Reward should remain at 1.0
        env.rovers()[0].set_position(45.0, 25.0)
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        
        # -- Rover goes to poi A, then poi B, then returns to the start
        config = get_default_config(self)
        new_poi_config = self.get_default_poi_config()
        new_poi_config['position']['fixed'] = [10.0, 25.0]
        config['env']['pois']['rover_pois'].append(
            new_poi_config
        )
        env = createEnv(config)
        _, (reward) = env.reset()
        # Rover starts far from poi. No reward
        expected_reward = 0.0
        self.assert_np_close(reward, expected_reward)
        # Rover moves nearby first poi. Rewarded for that poi
        env.rovers()[0].set_position(28.0, 25.0)
        rover_position = [env.rovers()[0].position().x, env.rovers()[0].position().y]
        poi_position = [env.pois()[0].position().x, env.pois()[0].position().y]
        expected_reward = self.compute_poi_reward_using_positions(rover_position, poi_position)
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        # Rover moves on top of first poi. Reward is 1.0
        env.rovers()[0].set_position(25.0, 25.0)
        reward = self.compute_G(env)
        expected_reward = 1.0
        # Rover moves closer to second poi. 
        # Reward of 1.0 for first poi, and partial reward for second poi
        env.rovers()[0].set_position(13.5, 25.0)
        rover_position = [env.rovers()[0].position().x, env.rovers()[0].position().y]
        poi_position = [env.pois()[1].position().x, env.pois()[1].position().y]
        expected_reward = 1.0+self.compute_poi_reward_using_positions(rover_position, poi_position)
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        # Rover moves on top of second poi. Reward of 2.0, 1.0 for each poi
        env.rovers()[0].set_position(10.0, 25.0)
        expected_reward = 2.0
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)
        # Rover moves back to its starting point. Reward should remain the same
        env.rovers()[0].set_position(45.0, 25.0)
        reward = self.compute_G(env)
        self.assert_np_close(reward, expected_reward)

        # -- Rover A goes to a poi, then leaves. Rover B gets closer to that poi, then leaves.
        config = get_default_config(self)
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        env = createEnv(config)
        # Neither rover has observed a poi, so no reward at the start
        _, (reward, _) = env.reset()
        expected_reward = 0.0
        self.assert_np_close(reward, expected_reward)
        # Rover A gets within the poi's observation radius. Rover B stays still. Rewards go up
        env.rovers()[0].set_position(28.0, 25.0)
        env.rovers()[1].set_position(env.rovers()[1].position().x, env.rovers()[1].position().y)
        expected_rewards = [self.compute_poi_reward(env.pois()[0], env.rovers()[0])]*2
        rewards = [self.compute_agent_reward(env, agent_id=i) for i in range(len(env.rovers()))]
        self.assert_close_lists(rewards, expected_rewards)
        # Rover A leaves. Rover B stays still. Rewards remain the same as before
        env.rovers()[0].set_position(45.0, 25.0)
        env.rovers()[1].set_position(env.rovers()[1].position().x, env.rovers()[1].position().y)
        rewards = [self.compute_agent_reward(env, agent_id=i) for i in range(len(env.rovers()))]
        self.assert_close_lists(rewards, expected_rewards)
        # Rover A stays still. Rover B gets closer to the POI than A got. Rewards go up
        env.rovers()[0].set_position(env.rovers()[0].position().x, env.rovers()[0].position().y)
        env.rovers()[1].set_position(27.0, 25.0)
        expected_rewards = [self.compute_poi_reward(env.pois()[0], env.rovers()[1])]
        rewards = [self.compute_agent_reward(env, agent_id=i) for i in range(len(env.rovers()))]
        self.assert_close_lists(rewards, expected_rewards)
        # Rover A stays still. Rover B leaves the POI. Rewards remain the same
        env.rovers()[0].set_position(env.rovers()[0].position().x, env.rovers()[0].position().y)
        env.rovers()[1].set_position(45.0, 25.0)
        rewards = [self.compute_agent_reward(env, agent_id=i) for i in range(len(env.rovers()))]

    def test_D(self):
        """Run some simple checks to make sure D works properly
        - 1 rover, 1 POI. D is the same as G
        - 2 rovers, 1 POI. The closer rover gets G through D. The further rover gets rewarded 0
        - 2 rovers, 2 POIs. Each rover gets rewarded for its POI
        - 4 rovers, 4 POIs. Each rover gets rewarded for its POI
        """
        self.default_poi_config['observation_radius'] = 5.0
        def get_default_config(self):
            config = self.get_env_template_config()
            config['env']['map_size'] = [50., 50.]
            return config
        # -- 1 rover, 1 POI. D is the same as G
        config = get_default_config(self)
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [10.0, 10.0]
        config['env']['agents']['rovers'] = [deepcopy(rover_config)]
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [10.0, 10.0]
        config['env']['pois']['rover_pois'] = [deepcopy(poi_config)]
        # Check with G
        self.assert_correct_rewards(config, expected_rewards=[1.0])
        # Check with D
        config['env']['agents']['rovers'][0]['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0])
        # -- 2 rovers, 1 POI. The closer rover gets G through D. The further rover gets rewarded 0
        # Add a rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [40.0, 40.0]
        rover_config['reward_type'] = 'Difference'
        config['env']['agents']['rovers'].append(deepcopy(rover_config))
        # Check with D for both rovers
        self.assert_correct_rewards(config, expected_rewards=[1.0, 0.0])
        # Now check with G for both rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Global'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
        # -- 2 rovers, 2 POIs. Each rover gets rewarded for its POI
        # Add a POI at the second rover
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [40.0, 40.0]
        config['env']['pois']['rover_pois'].append(deepcopy(poi_config))
        # Check with G for both rovers
        self.assert_correct_rewards(config, expected_rewards=[2.0, 2.0])
        # Check with D for both rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])
        # -- 4 rovers, 4 POIs. Each rover gets rewarded for its POI
        # Switch our current rovers back to G
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Global'
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
        # Check G for all 4 rovers
        self.assert_correct_rewards(config, expected_rewards=[4.0]*4)
        # Switch rovers to D and check D for all 4 rovers
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Difference'
        self.assert_correct_rewards(config, expected_rewards=[1.0]*4)

    def test_sequential_D(self):
        """Run some simple checks to make sure D works properly when G is sequential
        - 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
        - 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
            D should update accordingly
        - 2 rovers, 2 POIs. Each rover visits its respective POI.
        - 2 rovers, 2 POIs. Each rover visits both POIs.
        """
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'
        self.default_poi_config['position']['fixed'] = [10.0, 10.0]
        self.default_rover_config['position']['fixed'] = [40.0, 10.0]
        def get_default_config(self):
            config = self.get_env_template_config()
            config['env']['map_size'] = [50., 50.]
            return config
        # -- 1 rover, 1 POI. Rover goes to POI then leaves. D is the same as G.
        # Run this with G
        config = get_default_config(self)
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config()]
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
        # -- 2 rovers, 1 POI. Rover A goes to POI then leaves. Rover B goes to POI at a later time.
        # Switch rover back to G
        config['env']['agents']['rovers'][0]['reward_type'] = 'Global'
        # Add a rover to config
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
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
        # -- 2 rovers, 2 POIs. Each rover visits its respective POI.
        # Switch rovers back to G
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Global'
        # Add a poi in a different location
        poi_config = self.get_default_poi_config()
        poi_config['position']['fixed'] = [10.0, 20.0]
        config['env']['pois']['rover_pois'].append(poi_config)
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
        # -- 2 rovers, 2 POIs. Each rover visits both POIs.
        # Switch rovers back to G
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_type'] = 'Global'
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

# TODO: Write a test case for D that uses rovers and uavs together. I suspect something will go wrong
# with rovers and uavs BECAUSE the custom reward structure does not account for an agent being
# removed when checking agent_types
# (to check if agent is rover or uav, which determines if it can get rewarded for going to POIs)

if __name__ == '__main__':
    unittest.main()
