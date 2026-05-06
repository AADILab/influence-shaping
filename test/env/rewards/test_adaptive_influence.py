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

    def get_one_rover_three_uavs_one_poi_config_c(self):
        # 1 rover, 3 uavs, 1 POI. UAVs influence rover sequentially.
        self.default_poi_config['observation_radius'] = 1.0
        self.default_poi_config['constraint'] = 'sequential'

        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 10.]
        config['env']['agents']['rovers'] = [self.get_default_rover_config()]
        config['env']['agents']['uavs'] = [
            self.get_default_uav_config(),  # uav A
            self.get_default_uav_config(),  # uav B
            self.get_default_uav_config(),  # uav C
        ]
        config['env']['pois']['rover_pois'] = [self.get_default_poi_config()]

        # Rover starts far left, POI is far right
        config['env']['agents']['rovers'][0]['position']['fixed'] = [0.0, 5.0]
        # UAVs start at different positions along the map
        config['env']['agents']['uavs'][0]['position']['fixed'] = [0.0,  5.0]  # uav A starts with rover
        config['env']['agents']['uavs'][1]['position']['fixed'] = [20.0, 5.0]  # uav B starts in middle
        config['env']['agents']['uavs'][2]['position']['fixed'] = [35.0, 5.0]  # uav C starts near POI
        config['env']['pois']['rover_pois'][0]['position']['fixed'] = [49.0, 5.0]

        return config

    def get_path_c1(self):
        # [rover, uav A, uav B, uav C]
        # Each uav influences the rover for a distinct window of time, sequentially.
        # uav A:  t=0  to t=9  (first 10 steps)
        # uav B:  t=10 to t=19 (middle 10 steps)
        # uav C:  t=20 to t=29 (last 10 steps, leads rover to POI)
        # Rover reaches POI at t=29
        return [
            # t=0: uav A is with rover. uav B and C stay put.
            [[ 0.0, 5.0], [ 0.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[ 2.0, 5.0], [ 2.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[ 4.0, 5.0], [ 4.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[ 6.0, 5.0], [ 6.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[ 8.0, 5.0], [ 8.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[10.0, 5.0], [10.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[12.0, 5.0], [12.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[14.0, 5.0], [14.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[16.0, 5.0], [16.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[18.0, 5.0], [18.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            # t=10: uav A leaves, uav B takes over. uav A stays put, uav C stays put.
            [[20.0, 5.0], [18.0, 5.0], [20.0, 5.0], [35.0, 5.0]],
            [[22.0, 5.0], [18.0, 5.0], [22.0, 5.0], [35.0, 5.0]],
            [[24.0, 5.0], [18.0, 5.0], [24.0, 5.0], [35.0, 5.0]],
            [[26.0, 5.0], [18.0, 5.0], [26.0, 5.0], [35.0, 5.0]],
            [[28.0, 5.0], [18.0, 5.0], [28.0, 5.0], [35.0, 5.0]],
            [[30.0, 5.0], [18.0, 5.0], [30.0, 5.0], [35.0, 5.0]],
            [[32.0, 5.0], [18.0, 5.0], [32.0, 5.0], [35.0, 5.0]],
            [[34.0, 5.0], [18.0, 5.0], [34.0, 5.0], [35.0, 5.0]],
            [[36.0, 5.0], [18.0, 5.0], [36.0, 5.0], [35.0, 5.0]],
            [[38.0, 5.0], [18.0, 5.0], [38.0, 5.0], [35.0, 5.0]],
            # t=20: uav B leaves, uav C takes over. uav A and B stay put.
            [[40.0, 5.0], [18.0, 5.0], [38.0, 5.0], [40.0, 5.0]],
            [[42.0, 5.0], [18.0, 5.0], [38.0, 5.0], [42.0, 5.0]],
            [[44.0, 5.0], [18.0, 5.0], [38.0, 5.0], [44.0, 5.0]],
            [[46.0, 5.0], [18.0, 5.0], [38.0, 5.0], [46.0, 5.0]],
            [[48.0, 5.0], [18.0, 5.0], [38.0, 5.0], [48.0, 5.0]],
            [[49.0, 5.0], [18.0, 5.0], [38.0, 5.0], [49.0, 5.0]],
        ]

    def get_adaptive_config_c(self, config, N_agents, n_timesteps):
        # Helper to set all agents to adaptive influence with given params
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference': {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': N_agents,
                        'n_timesteps': n_timesteps
                    }
                }
            }
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference': {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': N_agents,
                        'n_timesteps': n_timesteps
                    }
                }
            }
        return config

    def test_one_rover_three_uavs_one_poi_config_c_path_c1_G(self):
        # Sanity check: G gives everyone equal reward
        config = self.get_one_rover_three_uavs_one_poi_config_c()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B, uav C
            [0.0, 0.0, 0.0, 0.0],  # t=0
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # t=10: uav B takes over
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # t=20: uav C takes over
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],  # t=25: rover reaches POI
        ]
        self.assert_path_rewards(env, self.get_path_c1(), expected_rewards_at_each_step)

    def test_one_rover_three_uavs_one_poi_config_c_path_c1_D(self):
        # Sanity check: D gives only rover credit
        config = self.get_one_rover_three_uavs_one_poi_config_c()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec']['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B, uav C
            [0.0, 0.0, 0.0, 0.0],  # t=0
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],  # t=25: only rover gets credit
        ]
        self.assert_path_rewards(env, self.get_path_c1(), expected_rewards_at_each_step)

    def test_one_rover_three_uavs_one_poi_config_c_path_c1_adaptive_N0_n0(self):
        # N=0, n=0: no extension. Only the directly influencing uav at POI time gets credit.
        # uav C is influencing the rover when it reaches the POI, so only uav C gets credit.
        config = self.get_one_rover_three_uavs_one_poi_config_c()
        config = self.get_adaptive_config_c(config, N_agents=0, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B, uav C
            *[[0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 0.0, 0.0, 1.0],  # t=25: only uav C gets credit
        ]
        self.assert_path_rewards(env, self.get_path_c1(), expected_rewards_at_each_step)

    def test_one_rover_three_uavs_one_poi_config_c_path_c1_adaptive_N1_n0(self):
        # N=1, n=0: uav A's influence extends until ONE other uav stops influencing the rover.
        # uav B starts and then stops influencing the rover before uav C arrives.
        # So uav A's influence should extend through uav B's window but NOT into uav C's window.
        # uav B's influence extends until uav C stops -- but uav C is still influencing at POI time,
        # so uav B's influence extends all the way to the POI.
        # uav C is directly influencing the rover at POI time, so it always gets credit.
        config = self.get_one_rover_three_uavs_one_poi_config_c()
        config = self.get_adaptive_config_c(config, N_agents=1, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B, uav C
            *[[0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 0.0, 1.0, 1.0],  # t=25: uav A does NOT get credit, uav B and C do
        ]
        self.assert_path_rewards(env, self.get_path_c1(), expected_rewards_at_each_step)

    def test_one_rover_three_uavs_one_poi_config_c_path_c1_adaptive_N2_n0(self):
        # N=2, n=0: uav A's influence extends until TWO other uavs have stopped influencing the rover.
        # uav B stops after its window, then uav C stops -- but uav C has not stopped by POI time.
        # So uav A's influence should extend through uav B's window and into uav C's window,
        # meaning uav A gets credit at POI time.
        # uav B similarly extends through uav C's window since only one agent (uav C) is left after B stops.
        # uav C is directly influencing, so always gets credit.
        config = self.get_one_rover_three_uavs_one_poi_config_c()
        config = self.get_adaptive_config_c(config, N_agents=2, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            # rover, uav A, uav B, uav C
            *[[0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 1.0, 1.0, 1.0],  # t=25: all uavs get credit
        ]
        self.assert_path_rewards(env, self.get_path_c1(), expected_rewards_at_each_step)

    def get_two_rover_six_uavs_two_pois_config_d(self):
        # 2 rovers, 6 uavs, 2 POIs. Bottom row mirrors config_c, top row is identical copy.
        self.default_poi_config['observation_radius'] = 1.0
        self.default_poi_config['constraint'] = 'sequential'

        config = self.get_env_template_config()
        config['env']['map_size'] = [50., 20.]
        config['env']['agents']['rovers'] = [
            self.get_default_rover_config(),  # rover bottom
            self.get_default_rover_config(),  # rover top
        ]
        config['env']['agents']['uavs'] = [
            self.get_default_uav_config(),  # uav A bottom
            self.get_default_uav_config(),  # uav B bottom
            self.get_default_uav_config(),  # uav C bottom
            self.get_default_uav_config(),  # uav A top
            self.get_default_uav_config(),  # uav B top
            self.get_default_uav_config(),  # uav C top
        ]
        config['env']['pois']['rover_pois'] = [
            self.get_default_poi_config(),  # poi bottom
            self.get_default_poi_config(),  # poi top
        ]

        # Bottom row (y=5) — mirrors config_c
        config['env']['agents']['rovers'][0]['position']['fixed']  = [ 0.0,  5.0]
        config['env']['agents']['uavs'][0]['position']['fixed']    = [ 0.0,  5.0]  # uav A bottom
        config['env']['agents']['uavs'][1]['position']['fixed']    = [20.0,  5.0]  # uav B bottom
        config['env']['agents']['uavs'][2]['position']['fixed']    = [35.0,  5.0]  # uav C bottom
        config['env']['pois']['rover_pois'][0]['position']['fixed'] = [49.0,  5.0]

        # Top row (y=15) — mirror of bottom row
        config['env']['agents']['rovers'][1]['position']['fixed']  = [ 0.0, 15.0]
        config['env']['agents']['uavs'][3]['position']['fixed']    = [ 0.0, 15.0]  # uav A top
        config['env']['agents']['uavs'][4]['position']['fixed']    = [20.0, 15.0]  # uav B top
        config['env']['agents']['uavs'][5]['position']['fixed']    = [35.0, 15.0]  # uav C top
        config['env']['pois']['rover_pois'][1]['position']['fixed'] = [49.0, 15.0]

        return config

    def get_path_d1(self):
        # Only bottom rover is guided to its POI. Top rover and its uavs stay put.
        # [rover bot, rover top, uav A bot, uav B bot, uav C bot, uav A top, uav B top, uav C top]
        return [
            # t=0-9: uav A bottom guides rover bottom
            [[ 0.0,  5.0], [ 0.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 2.0,  5.0], [ 0.0, 15.0], [ 2.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 4.0,  5.0], [ 0.0, 15.0], [ 4.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 6.0,  5.0], [ 0.0, 15.0], [ 6.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 8.0,  5.0], [ 0.0, 15.0], [ 8.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[10.0,  5.0], [ 0.0, 15.0], [10.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[12.0,  5.0], [ 0.0, 15.0], [12.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[14.0,  5.0], [ 0.0, 15.0], [14.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[16.0,  5.0], [ 0.0, 15.0], [16.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[18.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            # t=10-19: uav B bottom guides rover bottom. uav A bottom stays put.
            [[20.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[22.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [22.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[24.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [24.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[26.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [26.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[28.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [28.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[30.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [30.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[32.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [32.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[34.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [34.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[36.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [36.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[38.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            # t=20-24: uav C bottom guides rover bottom. uav A and B bottom stay put.
            [[40.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [40.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[42.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [42.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[44.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [44.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[46.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [46.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[48.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [48.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[49.0,  5.0], [ 0.0, 15.0], [18.0,  5.0], [38.0,  5.0], [49.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
        ]

    def get_path_d2(self):
        # Only top rover is guided to its POI. Bottom rover and its uavs stay put.
        # [rover bot, rover top, uav A bot, uav B bot, uav C bot, uav A top, uav B top, uav C top]
        return [
            # t=0-9: uav A top guides rover top
            [[ 0.0,  5.0], [ 0.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [ 2.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 2.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [ 4.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 4.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [ 6.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 6.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [ 8.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 8.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [10.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [10.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [12.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [12.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [14.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [14.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [16.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [16.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [18.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            # t=10-19: uav B top guides rover top. uav A top stays put.
            [[ 0.0,  5.0], [20.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [22.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [22.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [24.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [24.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [26.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [26.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [28.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [28.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [30.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [30.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [32.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [32.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [34.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [34.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [36.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [36.0, 15.0], [35.0, 15.0]],
            [[ 0.0,  5.0], [38.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [35.0, 15.0]],
            # t=20-24: uav C top guides rover top. uav A and B top stay put.
            [[ 0.0,  5.0], [40.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [40.0, 15.0]],
            [[ 0.0,  5.0], [42.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [42.0, 15.0]],
            [[ 0.0,  5.0], [44.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [44.0, 15.0]],
            [[ 0.0,  5.0], [46.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [46.0, 15.0]],
            [[ 0.0,  5.0], [48.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [48.0, 15.0]],
            [[ 0.0,  5.0], [49.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [49.0, 15.0]],
        ]

    def get_path_d3(self):
        # Both rovers guided simultaneously to their respective POIs.
        # [rover bot, rover top, uav A bot, uav B bot, uav C bot, uav A top, uav B top, uav C top]
        return [
            [[ 0.0,  5.0], [ 0.0, 15.0], [ 0.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 0.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 2.0,  5.0], [ 2.0, 15.0], [ 2.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 2.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 4.0,  5.0], [ 4.0, 15.0], [ 4.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 4.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 6.0,  5.0], [ 6.0, 15.0], [ 6.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 6.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[ 8.0,  5.0], [ 8.0, 15.0], [ 8.0,  5.0], [20.0,  5.0], [35.0,  5.0], [ 8.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[10.0,  5.0], [10.0, 15.0], [10.0,  5.0], [20.0,  5.0], [35.0,  5.0], [10.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[12.0,  5.0], [12.0, 15.0], [12.0,  5.0], [20.0,  5.0], [35.0,  5.0], [12.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[14.0,  5.0], [14.0, 15.0], [14.0,  5.0], [20.0,  5.0], [35.0,  5.0], [14.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[16.0,  5.0], [16.0, 15.0], [16.0,  5.0], [20.0,  5.0], [35.0,  5.0], [16.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[18.0,  5.0], [18.0, 15.0], [18.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[20.0,  5.0], [20.0, 15.0], [18.0,  5.0], [20.0,  5.0], [35.0,  5.0], [18.0, 15.0], [20.0, 15.0], [35.0, 15.0]],
            [[22.0,  5.0], [22.0, 15.0], [18.0,  5.0], [22.0,  5.0], [35.0,  5.0], [18.0, 15.0], [22.0, 15.0], [35.0, 15.0]],
            [[24.0,  5.0], [24.0, 15.0], [18.0,  5.0], [24.0,  5.0], [35.0,  5.0], [18.0, 15.0], [24.0, 15.0], [35.0, 15.0]],
            [[26.0,  5.0], [26.0, 15.0], [18.0,  5.0], [26.0,  5.0], [35.0,  5.0], [18.0, 15.0], [26.0, 15.0], [35.0, 15.0]],
            [[28.0,  5.0], [28.0, 15.0], [18.0,  5.0], [28.0,  5.0], [35.0,  5.0], [18.0, 15.0], [28.0, 15.0], [35.0, 15.0]],
            [[30.0,  5.0], [30.0, 15.0], [18.0,  5.0], [30.0,  5.0], [35.0,  5.0], [18.0, 15.0], [30.0, 15.0], [35.0, 15.0]],
            [[32.0,  5.0], [32.0, 15.0], [18.0,  5.0], [32.0,  5.0], [35.0,  5.0], [18.0, 15.0], [32.0, 15.0], [35.0, 15.0]],
            [[34.0,  5.0], [34.0, 15.0], [18.0,  5.0], [34.0,  5.0], [35.0,  5.0], [18.0, 15.0], [34.0, 15.0], [35.0, 15.0]],
            [[36.0,  5.0], [36.0, 15.0], [18.0,  5.0], [36.0,  5.0], [35.0,  5.0], [18.0, 15.0], [36.0, 15.0], [35.0, 15.0]],
            [[38.0,  5.0], [38.0, 15.0], [18.0,  5.0], [38.0,  5.0], [35.0,  5.0], [18.0, 15.0], [38.0, 15.0], [35.0, 15.0]],
            [[40.0,  5.0], [40.0, 15.0], [18.0,  5.0], [38.0,  5.0], [40.0,  5.0], [18.0, 15.0], [38.0, 15.0], [40.0, 15.0]],
            [[42.0,  5.0], [42.0, 15.0], [18.0,  5.0], [38.0,  5.0], [42.0,  5.0], [18.0, 15.0], [38.0, 15.0], [42.0, 15.0]],
            [[44.0,  5.0], [44.0, 15.0], [18.0,  5.0], [38.0,  5.0], [44.0,  5.0], [18.0, 15.0], [38.0, 15.0], [44.0, 15.0]],
            [[46.0,  5.0], [46.0, 15.0], [18.0,  5.0], [38.0,  5.0], [46.0,  5.0], [18.0, 15.0], [38.0, 15.0], [46.0, 15.0]],
            [[48.0,  5.0], [48.0, 15.0], [18.0,  5.0], [38.0,  5.0], [48.0,  5.0], [18.0, 15.0], [38.0, 15.0], [48.0, 15.0]],
            [[49.0,  5.0], [49.0, 15.0], [18.0,  5.0], [38.0,  5.0], [49.0,  5.0], [18.0, 15.0], [38.0, 15.0], [49.0, 15.0]],
        ]

    def get_adaptive_config_d(self, config, N_agents, n_timesteps):
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference': {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': N_agents,
                        'n_timesteps': n_timesteps
                    }
                }
            }
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec'] = {
                'reward_type': 'IndirectDifference',
                'indirect_difference': {
                    'mode': 'Adaptive',
                    'adaptive': {
                        'N_agents': N_agents,
                        'n_timesteps': n_timesteps
                    }
                }
            }
        return config

    # ---- path d1: only bottom rover reaches POI ----

    def test_two_rover_six_uavs_two_pois_config_d_path_d1_G(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        env = createEnv(config)
        # [rover bot, rover top, uav A bot, uav B bot, uav C bot, uav A top, uav B top, uav C top]
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # t=25: bottom POI captured, G gives everyone credit
        ]
        self.assert_path_rewards(env, self.get_path_d1(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d1_D(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec']['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # t=25: only bottom rover gets credit
        ]
        self.assert_path_rewards(env, self.get_path_d1(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d1_adaptive_N0_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=0, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # Only uav C bot gets credit (directly influencing bottom rover at POI time)
            # Top uavs get no credit since top rover never reaches its POI
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
        self.assert_path_rewards(env, self.get_path_d1(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d1_adaptive_N1_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=1, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # uav B bot extends through uav C bot's window. uav A bot does not.
            # Top uavs get no credit since top rover never reaches its POI.
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        self.assert_path_rewards(env, self.get_path_d1(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d1_adaptive_N2_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=2, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # All bottom uavs get credit. Top uavs get no credit.
            [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        self.assert_path_rewards(env, self.get_path_d1(), expected_rewards_at_each_step)

    # ---- path d2: only top rover reaches POI ----

    def test_two_rover_six_uavs_two_pois_config_d_path_d2_G(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # t=25: top POI captured, G gives everyone credit
        ]
        self.assert_path_rewards(env, self.get_path_d2(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d2_D(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec']['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # t=25: only top rover gets credit
        ]
        self.assert_path_rewards(env, self.get_path_d2(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d2_adaptive_N0_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=0, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # Only uav C top gets credit. Bottom uavs get no credit.
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d2(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d2_adaptive_N1_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=1, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # uav B top extends through uav C top's window. uav A top does not.
            # Bottom uavs get no credit.
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d2(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d2_adaptive_N2_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=2, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # All top uavs get credit. Bottom uavs get no credit.
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d2(), expected_rewards_at_each_step)

    # ---- path d3: both rovers reach their POIs ----

    def test_two_rover_six_uavs_two_pois_config_d_path_d3_G(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # t=25: both POIs captured simultaneously
        ]
        self.assert_path_rewards(env, self.get_path_d3(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d3_D(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        for rover_config in config['env']['agents']['rovers']:
            rover_config['reward_spec']['reward_type'] = 'Difference'
        for uav_config in config['env']['agents']['uavs']:
            uav_config['reward_spec']['reward_type'] = 'Difference'
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # t=25: each rover gets its own credit only
        ]
        self.assert_path_rewards(env, self.get_path_d3(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d3_adaptive_N0_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=0, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # uav C bot gets credit for bottom rover, uav C top gets credit for top rover
            # No cross-contamination between top and bottom
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d3(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d3_adaptive_N1_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=1, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # uav B and C bot get credit for bottom rover
            # uav B and C top get credit for top rover
            # uav A bot and uav A top do not get credit
            [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d3(), expected_rewards_at_each_step)

    def test_two_rover_six_uavs_two_pois_config_d_path_d3_adaptive_N2_n0(self):
        config = self.get_two_rover_six_uavs_two_pois_config_d()
        config = self.get_adaptive_config_d(config, N_agents=2, n_timesteps=0)
        env = createEnv(config)
        expected_rewards_at_each_step = [
            *[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 25,
            # All uavs get credit for their respective rovers. No cross-contamination.
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        self.assert_path_rewards(env, self.get_path_d3(), expected_rewards_at_each_step)

# Make a config where uav A passes a rover to uav B, and make sure that it works
# Extend that to a case where A passes rover to B, passes to C. Include uav D but it should not get credit

if __name__ == '__main__':
    unittest.main()
