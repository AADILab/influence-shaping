import unittest
from copy import deepcopy
from pathlib import Path
import pandas as pd

from influence.config import load_config
from influence.testing import TestEnv
from influence.custom_env import createEnv

class TestResults(TestEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_quartz(self):
        # Load all the relevant resources
        top_dir = '~/influence-shaping/test/resources/results/10_29_2024/quartz/1_rover_1_uav/random_pois_10x10/IndirectDifferenceAutomatic'
        config = load_config(
            config_dir=Path(top_dir)/'config.yaml'
        )
        joint_traj = pd.read_csv(
            Path(top_dir)/'trial_0'/'gen_0'/'eval_team_6_joint_traj.csv'
        )

        # Modify the config so that pois are fixed according to where they were in the joint trajectory
        poi_positions = [
            [joint_traj['hidden_poi_'+str(i)+'_x'][0], joint_traj['hidden_poi_'+str(i)+'_y'][0]]
            for i in range(5)
        ]
        for poi_config, poi_position in zip(config['env']['pois']['hidden_pois'], poi_positions):
            poi_config['position']['spawn_rule'] = 'fixed'
            poi_config['position']['fixed'] = poi_position

        # Turn the joint traj into a path
        agent_paths = []
        for rover_x, rover_y, uav_x, uav_y in zip(joint_traj['rover_0_x'][1:], joint_traj['rover_0_y'][1:], joint_traj['uav_0_x'][1:], joint_traj['uav_0_y'][1:]):
            agent_paths.append([ [rover_x, rover_y], [uav_x, uav_y] ])

        # Get the expected final rewards from the fitness csv file
        fitness_df = pd.read_csv(Path(top_dir)/'trial_0'/'fitness.csv')
        expected_final_rewards = [fitness_df['team_6_rover_0'][0], fitness_df['team_6_uav_0'][0]]

        # Now run the check
        self.assert_final_rewards(createEnv(config), agent_paths, expected_final_rewards)


if __name__ == '__main__':
    unittest.main()
