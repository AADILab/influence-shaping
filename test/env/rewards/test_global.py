import unittest

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestGlobal(TestEnv):
    """Check that G is computed as expected for different cases
    - additional rovers with low coupling (if coupling is satisfied, addtl rovers shouldn't change anything)
    - poi observation radii
    - uavs in the environment (should not affect G)
    - more pois
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_uav_config['position']['fixed'] = [23.0, 24.0]

    def get_default_config(self):
        config = self.get_env_template_config()
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        config['env']['agents']['uavs'] = []
        config['env']['pois']['rover_pois'].append(self.get_default_poi_config())
        config['env']['map_size'] = [50., 50.]
        return config
    
    def get_two_rovers_config(self):
        config = self.get_default_config()
        config['env']['agents']['rovers'][0]['position']['fixed'] = [26.0, 24.0]
        config['env']['agents']['rovers'].append(self.get_default_rover_config())
        return config
    
    def get_one_rover_three_pois_config(self):
        config = self.get_default_config()
        more_poi_positions = [
                    [25.0, 35.0],
                    [35.0, 35.0]
                ]
        poi_config = self.get_default_poi_config()
        for position in more_poi_positions:
            poi_config['position']['fixed'] = position
            config['env']['pois']['rover_pois'].append(poi_config)
        return config
    
    def get_one_rover_two_uavs_three_pois_config(self):
        config = self.get_one_rover_three_pois_config()
        uav_config = self.get_default_uav_config()
        for poi_config in config['env']['pois']['rover_pois']:
            uav_config['position']['fixed'] = poi_config['position']['fixed']
            config['env']['agents']['uavs'].append(uav_config)
        return config

    def get_three_rovers_three_uavs_three_pois_config(self):
        config = self.get_one_rover_three_pois_config()
        # Now let's add rovers to actually observe the remaining pois
        rover_config = self.get_default_rover_config()
        for poi_config in config['env']['pois']['rover_pois']:
            rover_config['position']['fixed'] = poi_config['position']['fixed']
            config['env']['agents']['rovers'].append(rover_config)
        return config

    def test_one_rover_one_poi(self):
        # Check one rover, one poi. Reward of 1.0
        config = self.get_default_config()
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
        config = self.get_default_config()
        env = createEnv(config)
        _, _ = env.reset()
        for position in rover_path:
            env.rovers()[0].set_position(position[0], position[1])
        reward = self.compute_G(env)
        expected_reward = 0.5
        self.assert_np_close(reward, expected_reward)
    
    def test_two_rovers_one_poi(self):
        # Now let's add another copy of our rover to make sure we are not double counting
        config = self.get_two_rovers_config()
        self.assert_correct_rewards(config, expected_rewards=[1.0, 1.0])

    def test_twelve_rovers_one_poi(self):
        # Add another 10 and check again for good measure
        config = self.get_two_rovers_config()
        for _ in range(10):
            config['env']['agents']['rovers'].append(self.get_default_rover_config())
        self.assert_correct_rewards(config, expected_rewards=[1.0]*12)

    def test_one_rover_one_poi_sparse(self):
        # Reset the config, and bring down the observation radius for the poi
        # G should still be 1.0
        config = self.get_default_config()
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
    
    def test_one_rover_three_uavs_one_poi(self):
        # Let's set the default config and throw in some uavs. Should not change G
        config = self.get_default_config()
        expected_reward = 1.0
        for _ in range(3):
            config['env']['agents']['uavs'].append(self.get_default_uav_config())
            env = createEnv(config)
            _, rewards = env.reset()
            reward = rewards[0]
            self.assert_np_close(reward, expected_reward)
    
    def test_one_rover_three_pois(self):
        # Let's set the default config and add more pois.
        config = self.get_one_rover_three_pois_config()
        # With only 1 rover, G should be 1.0 + the value of the other two pois
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 1.0 + \
            self.compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            self.compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assert_np_close(reward, expected_reward)

    def test_one_rover_two_uavs_three_pois(self):
        config = self.get_one_rover_two_uavs_three_pois_config()
        # Let's add uavs to observe the remaining pois 
        # (reward should not change because uavs can't observe pois)
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 1.0 + \
            self.compute_poi_reward(env.pois()[1], env.rovers()[0]) + \
            self.compute_poi_reward(env.pois()[2], env.rovers()[0])
        self.assert_np_close(reward, expected_reward)

    def test_three_rovers_three_uavs_three_pois(self):
        # Now let's add rovers to actually observe the remaining pois
        config = self.get_three_rovers_three_uavs_three_pois_config()
        env = createEnv(config)
        _, _ = env.reset()
        reward = self.compute_G(env)
        expected_reward = 3.0
        self.assert_np_close(reward, expected_reward)


if __name__ == '__main__':
    unittest.main()
