import unittest
import numpy as np
from copy import deepcopy
from influence.librovers import rovers
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
        print("TestEnv.assert_correct_rewards()")
        env = createEnv(config)
        _, rewards = env.reset()
        print(f"rewards, expected_rewards | {rewards}, {expected_rewards}")
        self.assert_close_lists(rewards, expected_rewards)

    def assert_close_lists(self, list0, list1):
        for element0, element1 in zip(list0, list1):
            self.assert_np_close(element0, element1)
