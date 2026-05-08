import unittest
from copy import deepcopy

from influence.testing import TestEnv
from influence.custom_env import createEnv


_GLOBAL_REWARD_SPEC = {'reward_type': 'Global'}
_DIFFERENCE_REWARD_SPEC = {'reward_type': 'Difference'}
_ALL_OR_NOTHING_REWARD_SPEC = {
    'reward_type': 'IndirectDifference',
    'indirect_difference': {
        'mode': 'Static',
        'static': {
            'assignment': 'automatic',
            'automatic': 'WinnerTakesAll',
        },
    },
}
_LOCAL_REWARD_SPEC = {
    'reward_type': 'IndirectDifference',
    'indirect_difference': {
        'mode': 'Static',
        'static': {
            'assignment': 'automatic',
            'automatic': 'Local',
        },
    },
}


class TestLooseStaticInfluence(TestEnv):
    """
    Tests for static loose (Local) vs. competitive (WinnerTakesAll) influence.

    Layout
    ------
    Config A — 1 rover, 3 UAVs, 1 POI
        Map: [60.0, 20.0]
        Agents (order in env): [Rover0, UAV0, UAV1, UAV2]
        Spawn: Rover0=(10,9), UAV0=(10,10), UAV1=(30,10), UAV2=(50,10)
        POI0=(50,10), obs_radius=5.0, constraint='sequential'

        Path A (35 steps):
          t=0       spawn: UAV0 is dist=1 from Rover0, UAV1/UAV2 are far away
          t=1..10    no movement — UAV0 keeps influencing Rover0 (total t=0..10 = 11 steps)
          t=11..20  Rover0 ->(30,9) — UAV1 now influences Rover0 (10 steps)
          t=21..35  Rover0 ->(50,9)— UAV2 now influences Rover0 (15 steps)

        Influence counts for Static Influence at t=21:
          UAV0: 11 steps (t=0..10), UAV1: 10 steps (t=11..20), UAV2: 1 step (t=21)
          → t=21..30: UAV0 wins (tie with UAV1, lower index wins with strict >)
          → t=31: UAV0 ties with UAV2 and wins by lower index
          → t=32..35: UAV2 wins (12 steps > UAV0's 11)

    Config B — 2 rovers, 6 UAVs, 2 POIs
        Map: [60.0, 40.0]
        Agents: [Rover0, Rover1, UAV0, UAV1, UAV2, UAV3, UAV4, UAV5]
        Spawn:
          Rover0=(10,9),  UAV0=(10,10),  UAV1=(30,10),  UAV2=(50,10)   ← lane A
          Rover1=(10,29), UAV3=(10,30), UAV4=(30,30), UAV5=(50,30)     ← lane B
        POI0=(50,10), POI1=(50,30), obs_radius=5.0, constraint='sequential'
        Lane separation is 20 units, preventing cross-influence (threshold=5).

        Path B1: both rovers reach their POIs at t=21 (mirror of Config A, both lanes)
        Path B2: Rover0 stays at spawn (10,9); only Rover1 moves
    """

    # ------------------------------------------------------------------ #
    #  Config A helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_config_a(self, reward_spec):
        config = self.get_env_template_config()
        config['env']['map_size'] = [60.0, 20.0]

        rover = self.get_default_rover_config()
        rover['position']['fixed'] = [10.0, 9.0]
        rover['reward_spec'] = deepcopy(reward_spec)

        def uav(x, y):
            c = self.get_default_uav_config()
            c['position']['fixed'] = [x, y]
            c['reward_spec'] = deepcopy(reward_spec)
            return c

        poi = self.get_default_poi_config()
        poi['observation_radius'] = 5.0
        poi['constraint'] = 'sequential'
        poi['position']['fixed'] = [50.0, 10.0]

        config['env']['agents']['rovers'] = [rover]
        config['env']['agents']['uavs'] = [uav(10, 10), uav(30, 10), uav(50, 10)]
        config['env']['pois']['rover_pois'] = [poi]
        return config

    def _get_path_a(self):
        """35-step path for Config A.

        Order: [Rover0, UAV0, UAV1, UAV2]
          t=1..10   no movement (all at spawn)
          t=11..20 Rover0 moves to (30,9), UAVs stay
          t=21..35 Rover0 moves to (50,9), UAVs stay
        """
        # Agent order: [Rover0, UAV0, UAV1, UAV2]
        spawn = [ [10, 9], [10, 10], [30, 10], [50, 10] ]
        to_30 = [ [30, 9], [10, 10], [30, 10], [50, 10] ]
        to_50 = [ [50, 9], [10, 10], [30, 10], [50, 10] ]
        return [spawn] * 10 + [to_30] * 10 + [to_50] * 15

    # ------------------------------------------------------------------ #
    #  Config A tests                                                    #
    # ------------------------------------------------------------------ #

    def test_config_a_global_reward(self):
        env = createEnv(self._get_config_a(_GLOBAL_REWARD_SPEC))
        n = 4
        expected = [[0.0] * n] * 21 + [[1.0] * n] * 15
        self.assert_path_rewards(env, self._get_path_a(), expected,
                                 start_msg='Config A / Global: ')

    def test_config_a_difference_reward(self):
        env = createEnv(self._get_config_a(_DIFFERENCE_REWARD_SPEC))
        n = 4
        zeros = [0.0] * n
        expected = [zeros] * 21 + [[1.0, 0.0, 0.0, 0.0]] * 15
        self.assert_path_rewards(env, self._get_path_a(), expected,
                                 start_msg='Config A / Difference: ')

    def test_config_a_static_all_or_nothing(self):
        env = createEnv(self._get_config_a(_ALL_OR_NOTHING_REWARD_SPEC))
        n = 4
        zeros = [0.0] * n
        # t=21..30: UAV0 wins (tie with UAV1 at 10 steps each; lower index wins)
        # t=31..35: UAV2 wins (12 steps > UAV0's 11)
        expected = (
            [zeros] * 21
            + [[1.0, 1.0, 0.0, 0.0]] * 11
            + [[1.0, 0.0, 0.0, 1.0]] * 4
        )
        self.assert_path_rewards(env, self._get_path_a(), expected,
                                 start_msg='Config A / AllOrNothing: ')

    def test_config_a_static_local(self):
        env = createEnv(self._get_config_a(_LOCAL_REWARD_SPEC))
        n = 4
        # All three UAVs influenced rover at some point → all get credit
        expected = [[0.0] * n] * 21 + [[1.0] * n] * 15
        self.assert_path_rewards(env, self._get_path_a(), expected,
                                 start_msg='Config A / Local: ')

    # ------------------------------------------------------------------ #
    #  Config B helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_config_b(self, reward_spec):
        config = self.get_env_template_config()
        config['env']['map_size'] = [60.0, 40.0]

        def rover(x, y):
            c = self.get_default_rover_config()
            c['position']['fixed'] = [x, y]
            c['reward_spec'] = deepcopy(reward_spec)
            return c

        def uav(x, y):
            c = self.get_default_uav_config()
            c['position']['fixed'] = [x, y]
            c['reward_spec'] = deepcopy(reward_spec)
            return c

        def poi(x, y):
            c = self.get_default_poi_config()
            c['observation_radius'] = 5.0
            c['constraint'] = 'sequential'
            c['position']['fixed'] = [x, y]
            return c

        config['env']['agents']['rovers'] = [rover(10.0, 9.0), rover(10.0, 29.0)]
        config['env']['agents']['uavs'] = [uav(10, 10), uav(30, 10), uav(50, 10), uav(10,30), uav(30, 30), uav(50, 30)]
        config['env']['pois']['rover_pois'] = [poi(50.0, 10.0), poi(50.0, 30.0)]
        return config

    def _get_path_b1(self):
        """35-step path — both rovers reach their POIs at t=21.

        Order: [Rover0, Rover1, UAV0, UAV1, UAV2, UAV3, UAV4, UAV5]
          t=1...10  no movement  (UAV0 influences Rover0, UAV3 influences Rover1)
          t=11..20  Rover0 moves to (30,9). Rover1 moves to (30,29), UAVs stay
          t=21..35  Rover0 moves to (50,9). Rover1 moves to (50,29), UAVs stay
        """
        # Agent order: [Rover0, Rover1, UAV0, UAV1, UAV2, UAV3, UAV4, UAV5]
        spawn = [
            [10, 9],  [10, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        to_30 = [
            [30, 9],  [30, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        to_50 = [
            [50, 9],  [50, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        return [spawn] * 10 + [to_30] * 10 + [to_50] * 15

    def _get_path_b2(self):
        """35-step path — Rover0 stays at spawn (10, 9); only Rover1 moves.

        Rover0 stays at (10, 9)
        Rover1 moves from (10, 29) to (30, 29), then (50, 29)
        """
        # Agent order: [Rover0, Rover1, UAV0, UAV1, UAV2, UAV3, UAV4, UAV5]
        spawn = [
            [10, 9],  [10, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        to_30 = [
            [10, 9],  [30, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        to_50 = [
            [10, 9],  [50, 29], # Rovers
            [10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30] # UAVs
        ]
        return [spawn] * 10 + [to_30] * 10 + [to_50] * 15

    # ------------------------------------------------------------------ #
    #  Config B Path B1 tests                                             #
    # ------------------------------------------------------------------ #

    def test_config_b_path_b1_global_reward(self):
        env = createEnv(self._get_config_b(_GLOBAL_REWARD_SPEC))
        n = 8
        expected = [[0.0] * n] * 21 + [[2.0] * n] * 15
        self.assert_path_rewards(env, self._get_path_b1(), expected,
                                 start_msg='Config B / B1 / Global: ')

    def test_config_b_path_b1_difference_reward(self):
        env = createEnv(self._get_config_b(_DIFFERENCE_REWARD_SPEC))
        n = 8
        zeros = [0.0] * n
        expected = [zeros] * 21 + [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 15
        self.assert_path_rewards(env, self._get_path_b1(), expected,
                                 start_msg='Config B / B1 / Difference: ')

    def test_config_b_path_b1_static_all_or_nothing(self):
        env = createEnv(self._get_config_b(_ALL_OR_NOTHING_REWARD_SPEC))
        n = 8
        zeros = [0.0] * n
        # t=21..31: UAV0 wins Rover0 (tie w/ UAV1), UAV3 wins Rover1 (tie w/ UAV4)
        # t=32..35: UAV2 wins Rover0, UAV5 wins Rover1 (12 influence steps for UAV2 > 11 for UAV5)
        expected = (
            [zeros] * 21
            + [[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 11
            + [[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]] * 4
        )
        self.assert_path_rewards(env, self._get_path_b1(), expected,
                                 start_msg='Config B / B1 / AllOrNothing: ')

    def test_config_b_path_b1_static_local(self):
        env = createEnv(self._get_config_b(_LOCAL_REWARD_SPEC))
        n = 8
        # All 6 UAVs influenced some rover at some point → all get credit
        expected = [[0.0] * n] * 21 + [[1.0] * n] * 15
        self.assert_path_rewards(env, self._get_path_b1(), expected,
                                 start_msg='Config B / B1 / Local: ')

    # ------------------------------------------------------------------ #
    #  Config B Path B2 tests                                             #
    # ------------------------------------------------------------------ #

    def test_config_b_path_b2_global_reward(self):
        env = createEnv(self._get_config_b(_GLOBAL_REWARD_SPEC))
        n = 8
        expected = [[0.0] * n] * 21 + [[1.0] * n] * 15
        self.assert_path_rewards(env, self._get_path_b2(), expected,
                                 start_msg='Config B / B2 / Global: ')

    def test_config_b_path_b2_difference_reward(self):
        env = createEnv(self._get_config_b(_DIFFERENCE_REWARD_SPEC))
        n = 8
        zeros = [0.0] * n
        # Rover0 never reaches POI0; only Rover1 contributes to G=1
        expected = [zeros] * 21 + [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 15
        self.assert_path_rewards(env, self._get_path_b2(), expected,
                                 start_msg='Config B / B2 / Difference: ')

    def test_config_b_path_b2_static_all_or_nothing(self):
        env = createEnv(self._get_config_b(_ALL_OR_NOTHING_REWARD_SPEC))
        n = 8
        zeros = [0.0] * n
        # t=21..31: UAV3 wins Rover1 (tie w/ UAV4, lower index)
        # t=32..35: UAV5 wins Rover1 (12 > 11)
        expected = (
            [zeros] * 21
            + [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 11
            + [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 4
        )
        self.assert_path_rewards(env, self._get_path_b2(), expected,
                                 start_msg='Config B / B2 / AllOrNothing: ')

    def test_config_b_path_b2_static_local(self):
        env = createEnv(self._get_config_b(_LOCAL_REWARD_SPEC))
        n = 8
        zeros = [0.0] * n
        # UAV0/UAV1 influenced only Rover0 (no POI) → reward=0
        # UAV2 never close enough to any rover → reward=0
        # UAV3/UAV4/UAV5 all influenced Rover1 at some point → reward=1
        expected = (
            [zeros] * 21
            + [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 15
        )
        self.assert_path_rewards(env, self._get_path_b2(), expected,
                                 start_msg='Config B / B2 / Local: ')


if __name__ == '__main__':
    unittest.main()
