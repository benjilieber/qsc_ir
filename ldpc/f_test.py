import unittest

import numpy as np

from ldpc.f import F


class FTest(unittest.TestCase):
    def test_init(self):
        f = F(3, 0.01, [1, 2, 0])
        np.testing.assert_array_equal(f.data, [[0.495, 0.01, 0.495], [0.495, 0.495, 0.01], [0.01, 0.495, 0.495]])

    def test_fork(self):
        forked_values = [0, 2]
        original_f = F(3, 0.01, [1, 2, 0])
        forked_f_list = original_f.fork(1, forked_values)
        assert(len(forked_f_list) == 2)
        for forked_value, forked_f in zip(forked_values, forked_f_list):
            forked_index_dist = [1.0, 0.0, 0.0] if forked_value == 0 else [0.0, 0.0, 1.0]
            np.testing.assert_array_equal(forked_f.data, [[0.495, 0.01, 0.495], forked_index_dist, [0.01, 0.495, 0.495]])


if __name__ == '__main__':
    unittest.main()
