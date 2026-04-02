import unittest

from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestAdaptive(TestEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_one_rover_one_uav_one_poi_config_a(self):
        # Set some defaults for the remainder of config
        self.default_poi_config['observation_radius'] = 5.0
        self.default_poi_config['constraint'] = 'sequential'
        self.default_poi_config['position']['fixed'] = [10.0, 10.0]
        self.default_rover_config['position']['fixed'] = [40.0, 10.0]
        self.default_uav_config['position']['fixed'] = [40.0, 10.0]

        # Get env template and fill it
        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 50.]
        # Throw in a rover
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        # Throw in a uav
        config['env']['agents']['uavs'] = [self.get_default_uav_config()]
        # Throw in a poi
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config()]
        return config

    def get_path_a1(self):
        # Used with config a
        return [
            # Rover A gets within the poi's observation radius. Uav stays at start location
            [[10.0, 10.0], [40.0, 10.0]]
        ]

    def get_path_a2(self):
        # Used with config a
        # Uav stays at start location through path
        return [
            # Rover goes away from the uav and sits there
            [[20.0, 10.0], [40.0, 10.0]],
            [[20.0, 10.0], [40.0, 10.0]],
            [[20.0, 10.0], [40.0, 10.0]],
            [[20.0, 10.0], [40.0, 10.0]],
            [[20.0, 10.0], [40.0, 10.0]],
            # Rover finally goes to POI
            [[10.0, 10.0], [40.0, 10.0]]
        ]

    def get_one_rover_two_uavs_two_pois_config_b(self):
        # Set some defaults for the remainder of config. POIs are small
        self.default_poi_config['observation_radius'] = 1.0
        self.default_poi_config['constraint'] = 'sequential'

        # Get env template and fill it
        config = self.get_env_template_config()
        config['env']['map_size'] = [20., 10.]
        # Throw in a rover
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        # Throw in 2 uavs
        config['env']['agents']['uavs'] = [self.get_default_uav_config(), self.get_default_uav_config()]
        # Throw in 2 pois
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config(), self.get_default_poi_config()]

        # Uavs are in bottom-left corners
        config['env']['agents']['uavs'][0]['position']['fixed'] = [0.0,  0.0]
        config['env']['agents']['uavs'][1]['position']['fixed'] = [10.0, 0.0]
        # Rover is with the first uav
        config['env']['agents']['rovers'][0]['position']['fixed'] = [0.0, 0.0]
        # POIs are in top-right corners
        config['env']['pois']['rover_pois'][0]['position']['fixed'] = [10.0, 10.0]
        config['env']['pois']['rover_pois'][1]['position']['fixed'] = [20.0, 10.0]

        return config

    def get_path_b1(self):
        # [ rover , uav A, uav B]
        return [
            # Uav A and rover A go to POI. Uav B stays still
            [[ 1.0,  1.0], [ 1.0,  1.0], [10.0,  0.0]],
            [[ 2.0,  2.0], [ 2.0,  2.0], [10.0,  0.0]],
            [[ 3.0,  3.0], [ 3.0,  3.0], [10.0,  0.0]],
            [[ 4.0,  4.0], [ 4.0,  4.0], [10.0,  0.0]],
            [[ 5.0,  5.0], [ 5.0,  5.0], [10.0,  0.0]],
            [[ 6.0,  6.0], [ 6.0,  6.0], [10.0,  0.0]],
            [[ 7.0,  7.0], [ 7.0,  7.0], [10.0,  0.0]],
            [[ 8.0,  8.0], [ 8.0,  8.0], [10.0,  0.0]],
            [[ 9.0,  9.0], [ 9.0,  9.0], [10.0,  0.0]],
            [[10.0, 10.0], [10.0, 10.0], [10.0,  0.0]],
            # Uav A and rover A go to Uav B. Uav B stays still
            [[10.0,  9.0], [10.0,  9.0], [10.0,  0.0]],
            [[10.0,  8.0], [10.0,  8.0], [10.0,  0.0]],
            [[10.0,  7.0], [10.0,  7.0], [10.0,  0.0]],
            [[10.0,  6.0], [10.0,  6.0], [10.0,  0.0]],
            [[10.0,  5.0], [10.0,  5.0], [10.0,  0.0]],
            [[10.0,  4.0], [10.0,  4.0], [10.0,  0.0]],
            [[10.0,  3.0], [10.0,  3.0], [10.0,  0.0]],
            [[10.0,  2.0], [10.0,  2.0], [10.0,  0.0]],
            [[10.0,  1.0], [10.0,  1.0], [10.0,  0.0]],
            [[10.0,  0.0], [10.0,  0.0], [10.0,  0.0]],
            # Uav B brings rover to second POI
            [[11.0,  1.0], [10.0,  0.0], [11.0,  1.0]],
            [[12.0,  2.0], [10.0,  0.0], [12.0,  2.0]],
            [[13.0,  3.0], [10.0,  0.0], [13.0,  3.0]],
            [[14.0,  4.0], [10.0,  0.0], [14.0,  4.0]],
            [[15.0,  5.0], [10.0,  0.0], [15.0,  5.0]],
            [[16.0,  6.0], [10.0,  0.0], [16.0,  6.0]],
            [[17.0,  7.0], [10.0,  0.0], [17.0,  7.0]],
            [[18.0,  8.0], [10.0,  0.0], [18.0,  8.0]],
            [[19.0,  9.0], [10.0,  0.0], [19.0,  9.0]],
            [[20.0, 10.0], [10.0,  0.0], [20.0, 10.0]]
        ]

    def test_one_rover_one_uav_one_poi_config_a_path_a1_G(self):
        # Simple 1-step sim where rover goes straight to POI
        # -- 1 rover, 1 uav, 1 POI. Rover goes to POI.
        # - First we try this with G
        config = self.get_one_rover_one_uav_one_poi_config_a()
        # Make the env and run it with G
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover moved onto POI. G is 1.0 for everyone.
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a1(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a1_adaptive_N0_n0(self):
        # - Next we do this with adaptive influence for the uav and D for the rover
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover moved onto POI. D is 1.0 for the rover, Uav gets no credit
            [1.0, 0.0]
        ]
        self.assert_path_rewards(env, self.get_path_a1(), expected_rewards_at_each_step)
        # Sanity check. This should also work if we give adaptive influence rewards to rover
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        self.assert_path_rewards(env, self.get_path_a1(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a1_adaptive_N0_n1(self):
        # - Modify adaptive influence to go up to 1 timestep extension
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 1
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover moved onto POI. D is 1.0 for the rover, Uav gets credit now
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a1(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a1_adaptive_N1_n0(self):
        # - Modify adaptive influence to go up to 1 agent, no timesteps
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 1,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover moved onto POI. D is 1.0 for the rover, Uav gets credit now
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a1(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_G(self):
        # Rover takes its time to get to the POI, so we need an extended influence window
        # -- 1 rover, 1 uav, 1 POI. Rover takes it time to get to a POI.
        config = self.get_one_rover_one_uav_one_poi_config_a()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover is not close enough to the POI
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            # Rover moved onto POI. G is 1.0 for everyone.
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_adaptive_N0_n0to5(self):
        # - Modify adaptive influence to go up incrementally 0-5 steps
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for n_timesteps in range(6):
            for uav_config in config['env']['agents']['uavs']:
                uav_config['reward_spec'] = {
                    'reward_type': 'IndirectDifference',
                    'indirect_difference' : {
                        'mode': 'Adaptive',
                        'adaptive': {
                            'N_agents': 0,
                            'n_timesteps': n_timesteps
                        }
                    }
                }
            env = createEnv(config)
            expected_rewards_at_each_step = [
                # Initial setup. No reward
                [0.0, 0.0],
                # Rover is not close enough to the POI
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                # Rover moved onto POI, but uav still gets no credit
                [1.0, 0.0]
            ]
            self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_adaptive_N0_n6(self):
        # Now we increment up to 6 steps, so uav should get credit now
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 6
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover is not close enough to the POI
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            # Rover moved onto POI, uav gets credit
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_adaptive_N0_n10(self):
        # Sanity check: Incrementing to 10 steps should not cause any issues
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 10
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover is not close enough to the POI
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            # Rover moved onto POI, uav gets credit
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_adaptive_N1_n0(self):
        # N=1 should also work here
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 1,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover is not close enough to the POI
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            # Rover moved onto POI, uav gets credit
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_one_uav_one_poi_config_a_path_a2_adaptive_N100_n0(self):
        # Sanity check: N=100 should be fine too
        config = self.get_one_rover_one_uav_one_poi_config_a()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 100,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0],
            # Rover is not close enough to the POI
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            # Rover moved onto POI, uav gets credit
            [1.0, 1.0]
        ]
        self.assert_path_rewards(env, self.get_path_a2(), expected_rewards_at_each_step)

    def test_one_rover_two_uavs_two_pois_config_b_path_b1_G(self):
        config = self.get_one_rover_two_uavs_two_pois_config_b()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # Initial setup. No reward
            [0.0, 0.0, 0.0],
            # Uav A and rover A go to POI. Uav B stays still
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            # Uav A and rover A go to Uav B
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            # Uav B brings rover to second POI
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
        self.assert_path_rewards(env, self.get_path_b1(), expected_rewards_at_each_step)

    def test_one_rover_two_uavs_two_pois_config_b_path_b1_D(self):
        # No credit for uavs
        config = self.get_one_rover_two_uavs_two_pois_config_b()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec']['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B
            # Initial setup. No reward
            [0.0, 0.0, 0.0],
            # Uav A and rover A go to POI. Uav B stays still
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            # Uav A and rover A go to Uav B
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            # Uav B brings rover to second POI
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
        self.assert_path_rewards(env, self.get_path_b1(), expected_rewards_at_each_step)

    def test_one_rover_two_uavs_two_pois_config_b_path_b1_adaptive_N0_n0(self):
        # Adaptive influence collapses into dynamic influence
        config = self.get_one_rover_two_uavs_two_pois_config_b()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 0
                    }
                }
            }
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 0,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B
            # Initial setup. No reward
            [0.0, 0.0, 0.0],
            # Uav A and rover A go to POI. Uav B stays still
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], # First POI captured
            # Uav A and rover A go to Uav B
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            # Uav B brings rover to second POI
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0] # Second POI captured
        ]
        self.assert_path_rewards(env, self.get_path_b1(), expected_rewards_at_each_step)

    def test_one_rover_two_uavs_two_pois_config_b_path_b1_adaptive_N1_n0(self):
        # Uav A gets credit for both POIs now
        config = self.get_one_rover_two_uavs_two_pois_config_b()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 1,
                        'n_timesteps': 0
                    }
                }
            }
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference' : {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': 1,
                        'n_timesteps': 0
                    }
                }
            }
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B
            # Initial setup. No reward
            [0.0, 0.0, 0.0],
            # Uav A and rover A go to POI. Uav B stays still
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], # First POI captured
            # Uav A and rover A go to Uav B
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            # Uav B brings rover to second POI
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0] # Second POI captured
        ]
        self.assert_path_rewards(env, self.get_path_b1(), expected_rewards_at_each_step)

# Make a config where uav A passes a rover to uav B, and make sure that it works
# Extend that to a case where A passes rover to B, passes to C. Include uav D but it should not get credit

if __name__ == '__main__':
    unittest.main()
