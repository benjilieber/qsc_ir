import math
import unittest

from key_generator import KeyGenerator
from protocol_configs import ProtocolConfigs
import numpy as np


class ProtocolConfigsTest(unittest.TestCase):
    def test_radius_for_max_block_error(self):
        block_size_list = list(range(3, 21))
        n = 10_000
        p_err = 0.01
        goal_p_bad = 0.99
        radii = [ProtocolConfigs(base = 3, block_length = block_size, num_blocks = n / block_size, p_err=p_err, success_rate=goal_p_bad).radius_for_max_block_error() for block_size in block_size_list]
        self.assertEqual([2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], radii)

    def test_overall_radius_key_error(self):
        n = 10_000
        block_length = 10
        p_err = 0.01
        goal_p_bad = 0.99
        radius = ProtocolConfigs(base = 3, block_length = block_length, num_blocks = n / block_length, p_err=p_err, success_rate=goal_p_bad).overall_radius_key_error()
        self.assertEqual(124, radius)

    def test_radii_for_goal_p_bad(self):
        n = 1000
        block_length = 8
        p_err = 0.05
        goal_p_bad = 0.99
        cfg = ProtocolConfigs(base=3, block_length=block_length, num_blocks=math.ceil(n / block_length), p_err=p_err, success_rate=goal_p_bad)
        keygen = KeyGenerator(p_err, n)
        cnt = 0
        num_runs = 100
        for _ in range(num_runs):
            a, b = keygen.generate_keys()
            cnt += cfg.is_within_radius_all_blocks(np.array_split(a, cfg.num_blocks), np.array_split(b, cfg.num_blocks))
        self.assertLessEqual(goal_p_bad, cnt/num_runs)

    def test_overall_prefixes_error_range(self):
        n = 10_000
        goal_p_bad = 0.99
        for block_length in range(8, 20):
            for p_err in np.arange(0.01, 0.11, 0.01):
                cfg = ProtocolConfigs(base=3, block_length=block_length, num_blocks=math.ceil(n / block_length), p_err=p_err, success_rate=goal_p_bad)
                print(cfg.prefix_radii)
                print(cfg.max_block_error)

    def test_timeout(self):
        timeout = 10
        cfg = ProtocolConfigs(base=3, block_length=8, num_blocks=1250,
                              p_err=0.01, success_rate=0.99, timeout=timeout)


if __name__ == '__main__':
    unittest.main()
