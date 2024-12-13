import unittest
import numpy as np
from copy import deepcopy
from typing import Any
from influence.librovers import rovers, eigen
from influence.custom_env import createEnv

class InfluenceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def assert_np_close(self, arg0, arg1):
        return self.assertTrue(np.isclose(arg0, arg1), f'{arg0} != {arg1}')

    @staticmethod
    def distance(entity_0, entity_1):
        return np.sqrt((entity_0.position().x - entity_1.position().x)**2 + (entity_0.position().y - entity_1.position().y)**2)
    
    @staticmethod
    def inverse_distance_squared(entity_0, entity_1):
        return 1. / InfluenceTestCase.distance(entity_0, entity_1)**2

    @staticmethod
    def compute_poi_reward(poi, agent):
        return 1.0 / max(InfluenceTestCase.distance(poi, agent), 1.0)
    
    @staticmethod
    def compute_poi_reward_using_positions(position0, position1):
        return 1.0 / max(np.linalg.norm(np.array(position0)-np.array(position1)), 1.0)
    
    @staticmethod
    def compute_G(env):
        return env.rovers()[0].reward(rovers.AgentPack(0, env.rovers(), env.pois()))

    @staticmethod
    def compute_agent_reward(env, agent_id):
        return env.rovers()[agent_id].reward(rovers.AgentPack(agent_id, env.rovers(), env.pois()))
    
    @staticmethod
    def extract_observation(cppyy_observation):
        return [cppyy_observation(i,0) for i in range(cppyy_observation.size())]

class TestEnv(InfluenceTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_rover_config = {
            'observation_radius': 1000.0,
            'position': {
                'spawn_rule': 'fixed',
                'fixed': [25.0, 24.0]
            },
            'resolution': 90,
            'reward_type': 'Global'
        }
        self.default_uav_config = {
            'observation_radius': 1000.0,
            'position': {
                'spawn_rule': 'fixed',
                'fixed': [26.0, 24.0]
            },
            'resolution': 90,
            'reward_type': 'Global'
        }
        self.default_poi_config = {
            'coupling': 1,
            'observation_radius': 1000.0,
            'position': {
                'spawn_rule': 'fixed',
                'fixed': [24.0, 24.0]
            },
            'value': 1.0
        }
    
    def get_default_rover_config(self):
        return deepcopy(self.default_rover_config)
    
    def get_default_uav_config(self):
        return deepcopy(self.default_uav_config)
    
    def get_default_poi_config(self):
        return deepcopy(self.default_poi_config)
    
    def get_env_template_config(self):
        return {
            'env': {
                'agents': {
                    'rovers': [],
                    'uavs': []
                },
                'pois': {
                    'rover_pois': [],
                    'hidden_pois': []
                }
            },
            'map_size': [0.0, 0.0]
        }

    def assert_correct_rewards(self, config, expected_rewards):
        # print("TestEnv.assert_correct_rewards()")
        env = createEnv(config)
        _, rewards = env.reset()
        # print(f"rewards, expected_rewards | {rewards}, {expected_rewards}")
        self.assert_close_lists(rewards, expected_rewards)

    def check_close_lists(self, list0, list1):
        for element0, element1 in zip(list0, list1):
            if not np.isclose(element0, element1):
                return False
        return True

    def assert_close_lists(self, list0, list1, msg: Any = None):
        # self.assertTrue(len(list0) == len(list1))
        if msg is None:
            msg = f'Elements in lists are not equal {list0} != {list1}'
        self.assertTrue(self.check_close_lists(list0, list1), msg)

    def rewards_from_env(self, env):
        # return [self.compute_agent_reward(env, agent_id=i) for i in range(len(env.rovers()))]
        # Need to ensure that rewards are now handled by the rewards computer 
        return env.status()[1]
    
    def get_agent_paths_from_env(self, env):
        raw_rover_paths = [rover.path() for rover in env.rovers()]
        paths = [[] for _ in range(env.rovers()[0].path().size())]
        # Each row is a timestep with the position of each rover at that timestep
        for raw_rover_path in raw_rover_paths:
            for t in range(raw_rover_path.size()):
                paths[t].append(self.position_as_list(raw_rover_path[t]))
        return paths

    def get_agent_positions_from_env(self, env):
        return [self.get_position_from_entity(agent) for agent in env.rovers()]
    
    def get_poi_positions_from_env(self, env):
        return [self.get_position_from_entity(poi) for poi in env.pois()]

    def get_position_from_entity(self, entity):
        return self.position_as_list(entity.position())

    def position_as_list(self, position_obj):
        return [position_obj.x, position_obj.y]

    def assert_path_rewards(self, env, agent_paths, expected_rewards_at_each_step, start_msg: Any = ''):
        # Reset up the env
        _, rewards = env.reset()
        # Make sure rewards at initial step are correct
        self.assert_close_lists(rewards, expected_rewards_at_each_step[0],
            msg=str(start_msg) + f'Rewards computed incorrectly at t=0\nExpected rewards: {expected_rewards_at_each_step[0]}\nEnv rewards: {rewards}')
        # Now loop through the paths and check rewards at each step
        for prev_t, (positions, expected_rewards) in enumerate(zip(agent_paths, expected_rewards_at_each_step[1:])):
            t = prev_t + 1
            # Set all the agent positions at this step
            for agent_id, (x,y) in enumerate(positions):
                env.rovers()[agent_id].set_position(x, y)
            # Check all agent rewards
            rewards = self.rewards_from_env(env)
            self.assert_close_lists(rewards, expected_rewards, 
                msg=str(start_msg)+f'Rewards computed incorrectly at t={t}\n' + \
                    f'Expected rewards: {expected_rewards}\n' + \
                    f'Env rewards: {rewards}\n' + \
                    f'Agent positions in env: {self.get_agent_positions_from_env(env)}\n' + \
                    f'Agent paths in env: {self.get_agent_paths_from_env(env)}\n' + \
                    f'POI positions in env: {self.get_poi_positions_from_env(env)}\n'
            )
