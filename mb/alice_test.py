import unittest

import numpy as np

from alice import Alice
from mb_cfg import MbCfg


class AliceTest(unittest.TestCase):

    def test_single_block_matrix(self):
        alice = Alice(MbCfg(q=3, num_blocks=3, block_length=3), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        encoding_matrices_delta, encoded_a = alice.encode_blocks([0], 2)
        self.assertEqual(encoding_matrices_delta.shape, (1, 3, 2))
        self.assertEqual(encoded_a.shape, (2,))
        self.assertTrue(np.isin(encoding_matrices_delta, [0, 1, 2]).all())
        self.assertTrue(np.isin(encoded_a, [0, 1, 2]).all())
        np.testing.assert_array_equal(np.matmul([0, 1, 2], encoding_matrices_delta[0]) % 3, encoded_a)

    def test_multi_block_matrix(self):
        alice = Alice(MbCfg(q=3, num_blocks=3, block_length=3), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        encoding_matrices_delta, encoded_a = alice.encode_blocks([0, 1], 2)
        self.assertEqual(encoding_matrices_delta.shape, (2, 3, 2))
        self.assertEqual(encoded_a.shape, (2,))
        self.assertTrue(np.isin(encoding_matrices_delta, [0, 1, 2]).all())
        self.assertTrue(np.isin(encoded_a, [0, 1, 2]).all())
        np.testing.assert_array_equal(
            (np.matmul([0, 1, 2], encoding_matrices_delta[0]) + np.matmul([2, 1, 2], encoding_matrices_delta[1])) % 3,
            encoded_a)

if __name__ == '__main__':
    unittest.main()
