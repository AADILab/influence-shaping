import unittest

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestSequentialGlobal(TestEnv):
    """Check that G is computed as expected when it is sequential
    - rover goes back and forth to and from a poi
    - rover goes to poi A, then poi B, then returns to starting point
    - rover A goes to poi, then leaves. rover B gets closer to poi, then leaves
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def test_one_rover_one_poi(self):
        # -- Rover goes back and forth to and from a poi
        config = self.get_default_config()
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
    
    def test_one_rover_two_pois(self):
        # -- Rover goes to poi A, then poi B, then returns to the start
        config = self.get_default_config()
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

    def test_two_rovers_one_poi(self):
        # -- Rover A goes to a poi, then leaves. Rover B gets closer to that poi, then leaves.
        config = self.get_default_config()
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

if __name__ == '__main__':
    unittest.main()
