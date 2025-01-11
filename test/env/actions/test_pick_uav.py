"""This is testing a new lidar for rovers where they sense their distance to all
uavs within their observation radius
"""

import unittest
import numpy as np
from influence.testing import TestEnv
from influence.custom_env import createEnv
from influence.ccea_lib import TeamInfo

class PickZeroPolicy():
    def forward(self):
        return np.array([0])

class DoNothingPolicy():
    def forward(self):
        return np.array([0., 0.])

class TestPickUav(TestEnv):
    def test_a(self):
        # Set up environment with rovers, uavs, and pois
        config = self.get_env_template_config()
        # Small map
        config['env']['map_size'] = [20., 20.]
        # One rover
        rover_config = self.get_default_rover_config()
        rover_config['position']['fixed'] = [5., 5.]
        config['env']['agents']['rovers'].append(rover_config)
        # one uav
        uav_config = self.get_default_uav_config()
        uav_config['position']['fixed'] = [10., 10.]
        config['env']['agents']['uavs'].append(uav_config)
        # Create the env
        env = createEnv(config)
        

if __name__ == '__main__':
    unittest.main()
