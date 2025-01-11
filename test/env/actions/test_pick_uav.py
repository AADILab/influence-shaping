"""This is testing a new lidar for rovers where they sense their distance to all
uavs within their observation radius
"""

import unittest
from copy import deepcopy
import numpy as np
from influence.testing import TestEnv
from influence.custom_env import createEnv
from influence.ccea_lib import TeamInfo, CooperativeCoevolutionaryAlgorithm, FollowPolicy

class PickZeroPolicy():
    def forward(self):
        return np.array([1,0])

class PickOnePolicy():
    def forward(self):
        return np.array([0,1])

class DoNothingPolicy():
    def forward(self):
        return np.array([0., 0.])

class TestPickUav(TestEnv):
    def test_a(self):
        """Rover goes to the only uav in the map"""
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        init_rover_position = [5., 5.]
        init_uav_position = [10., 10.]
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = deepcopy(init_rover_position)
        rover_config['action'] = {'type': 'pick_uav'}
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # Set velocity constraints
        config['ccea'] = {
            'network': {
                'rover_max_velocity': 1.0,
                'uav_max_velocity': 1.0
            }
        }
        
        # Build team policies
        # Rover always picks 0, uav does nothing
        policies = [PickZeroPolicy, DoNothingPolicy]
        team = TeamInfo(
            policies=policies,
            seed=None
        )
        
        # Run sim
        eval_info = CooperativeCoevolutionaryAlgorithm.evaluateTeamStatic(
            team=team,
            template_policies=policies,
            config=config,
            num_rovers=1,
            num_uavs=1,
            num_steps=10,
            compute_team_fitness=True
        )

        # The uav should have not moved and the rover should have moved at max velocity to the uav
        expected_rover_xs = np.linspace(5,10,6,dtype=int).tolist() + [10 for _ in range(4)]
        expected_rover_ys = expected_rover_xs
        expected_rover_positions = zip(
            expected_rover_xs,
            expected_rover_ys
        )
        for (rover_pos, uav_pos), expected_rover_pos in zip(eval_info.joint_trajectory.states, expected_rover_positions):
            self.assert_close_lists(rover_pos, expected_rover_pos)
            self.assert_close_lists(uav_pos, init_uav_position)

    def test_b(self):
        """Rover picks to follow no uavs"""
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        init_rover_position = [5., 5.]
        init_uav_position = [10., 10.]
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = deepcopy(init_rover_position)
        rover_config['action'] = {'type': 'pick_uav'}
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # Set velocity constraints
        config['ccea'] = {
            'network': {
                'rover_max_velocity': 1.0,
                'uav_max_velocity': 1.0
            }
        }

        # Build team policies
        # Rover always picks 1 (stay still), uav does nothing
        policies = [PickOnePolicy, DoNothingPolicy]
        team = TeamInfo(
            policies=policies,
            seed=None
        )
        
        # Run sim
        eval_info = CooperativeCoevolutionaryAlgorithm.evaluateTeamStatic(
            team=team,
            template_policies=policies,
            config=config,
            num_rovers=1,
            num_uavs=1,
            num_steps=10,
            compute_team_fitness=True
        )

        # The uav and rover both should not have moved
        for rover_pos, uav_pos in eval_info.joint_trajectory.states:
            self.assert_close_lists(rover_pos, init_rover_position)
            self.assert_close_lists(uav_pos, init_uav_position)

    def test_c(self):
        """Rover goes to the only uav in the map (This time using the follow policy)"""
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        init_rover_position = [5., 5.]
        init_uav_position = [10., 10.]
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = deepcopy(init_rover_position)
        rover_config['action'] = {'type': 'pick_uav'}
        rover_config['sensor'] = {'type': 'UavDistanceLidar'}
        rover_config['policy'] = {'type': 'follow'}
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # Set velocity constraints
        config['ccea'] = {
            'network': {
                'rover_max_velocity': 1.0,
                'uav_max_velocity': 1.0
            }
        }
        
        # Build team policies
        # Rover always picks 0, uav does nothing
        policies = [FollowPolicy, DoNothingPolicy]
        team = TeamInfo(
            policies=policies,
            seed=None
        )
        
        # Run sim
        eval_info = CooperativeCoevolutionaryAlgorithm.evaluateTeamStatic(
            team=team,
            template_policies=policies,
            config=config,
            num_rovers=1,
            num_uavs=1,
            num_steps=10,
            compute_team_fitness=True
        )

        # The uav should have not moved and the rover should have moved at max velocity to the uav
        expected_rover_xs = np.linspace(5,10,6,dtype=int).tolist() + [10 for _ in range(4)]
        expected_rover_ys = expected_rover_xs
        expected_rover_positions = zip(
            expected_rover_xs,
            expected_rover_ys
        )
        for (rover_pos, uav_pos), expected_rover_pos in zip(eval_info.joint_trajectory.states, expected_rover_positions):
            self.assert_close_lists(rover_pos, expected_rover_pos)
            self.assert_close_lists(uav_pos, init_uav_position)

    def test_d(self):
        """Rover chooses to follow the closest uav in the map. Rover does not follow the farther rover"""
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        init_rover_position = [5., 5.]
        init_uav_position_A = [10., 10.]
        init_uav_position_B = [15., 15.]
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = deepcopy(init_rover_position)
        rover_config['action'] = {'type': 'pick_uav'}
        rover_config['sensor'] = {'type': 'UavDistanceLidar'}
        rover_config['policy'] = {'type': 'follow'}
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position_A)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # second uav, further away
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position_B)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # Set velocity constraints
        config['ccea'] = {
            'network': {
                'rover_max_velocity': 1.0,
                'uav_max_velocity': 1.0
            }
        }
        
        # Build team policies
        # Rover always picks 0, uav does nothing
        policies = [FollowPolicy, DoNothingPolicy, DoNothingPolicy]
        team = TeamInfo(
            policies=policies,
            seed=None
        )
        
        # Run sim
        eval_info = CooperativeCoevolutionaryAlgorithm.evaluateTeamStatic(
            team=team,
            template_policies=policies,
            config=config,
            num_rovers=1,
            num_uavs=1,
            num_steps=10,
            compute_team_fitness=True
        )
        print(eval_info.joint_trajectory.states)

        # The uav should have not moved and the rover should have moved at max velocity to the uav
        expected_rover_xs = np.linspace(5,10,6,dtype=int).tolist() + [10 for _ in range(4)]
        expected_rover_ys = expected_rover_xs
        expected_rover_positions = zip(
            expected_rover_xs,
            expected_rover_ys
        )
        for (rover_pos, uav_pos_A, uav_pos_B), expected_rover_pos \
            in zip(eval_info.joint_trajectory.states, expected_rover_positions):
            self.assert_close_lists(rover_pos, expected_rover_pos)
            self.assert_close_lists(uav_pos_A, init_uav_position_A)
            self.assert_close_lists(uav_pos_B, init_uav_position_B)

    def test_e(self):
        """Rover chooses to stay still when it cannot see the uavs"""
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        init_rover_position = [5., 5.]
        init_uav_position_A = [10., 10.]
        init_uav_position_B = [15., 15.]
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = deepcopy(init_rover_position)
        rover_config['observation_radius'] = 1.0
        rover_config['action'] = {'type': 'pick_uav'}
        rover_config['sensor'] = {'type': 'UavDistanceLidar'}
        rover_config['policy'] = {'type': 'follow'}
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position_A)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # second uav, further away
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = deepcopy(init_uav_position_B)
        uav_config['action'] = {'type' : 'dxdy'}
        config['env']['agents']['uavs'].append(uav_config)
        # Set velocity constraints
        config['ccea'] = {
            'network': {
                'rover_max_velocity': 1.0,
                'uav_max_velocity': 1.0
            }
        }
        
        # Build team policies
        # Rover always picks 0, uav does nothing
        policies = [FollowPolicy, DoNothingPolicy, DoNothingPolicy]
        team = TeamInfo(
            policies=policies,
            seed=None
        )
        
        # Run sim
        eval_info = CooperativeCoevolutionaryAlgorithm.evaluateTeamStatic(
            team=team,
            template_policies=policies,
            config=config,
            num_rovers=1,
            num_uavs=len(config['env']['agents']['uavs']),
            num_steps=10,
            compute_team_fitness=True
        )

        for rover_pos, uav_pos_A, uav_pos_B in eval_info.joint_trajectory.states:
            self.assert_close_lists(rover_pos, init_rover_position)
            self.assert_close_lists(uav_pos_A, init_uav_position_A)
            self.assert_close_lists(uav_pos_B, init_uav_position_B)

if __name__ == '__main__':
    unittest.main()
