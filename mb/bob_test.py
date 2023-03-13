import unittest
from bob import Bob
from mb_cfg import MbCfg
import numpy as np


class BobTest(unittest.TestCase):

    def test_single_block(self):
        bob = Bob(MbCfg(q=3, num_blocks=3, block_length=3), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([0], np.array([[[0, 1], [1, 1], [2, 0]]]), np.array([2, 0]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0]]])
        self.assertEqual(cur_candidates_num, 1)
        self.assertEqual(num_candidates_per_block, [1])
        self.assertEqual(min_candidate_error, 0)


    def test_multi_block(self):
        bob = Bob(MbCfg(q=3, num_blocks=3, block_length=3), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        bob.decode_multi_block([0], np.array([[[0, 1], [1, 1], [2, 0]]]), np.array([2, 0]))
        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([1], np.array([[[2, 1], [1, 1], [0, 0]]]), np.array([0, 0]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0], [0, 0, 0]], [[1, 2, 0], [0, 0, 1]]])
        self.assertEqual(cur_candidates_num, 2)
        self.assertEqual(num_candidates_per_block, [1, 2])
        self.assertEqual(min_candidate_error, 0)

        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([1, 2], np.array([[[2, 0], [0, 1], [2, 1]], [[0, 1], [2, 1], [0, 2]]]), np.array([2, 1]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0], [0, 0, 0], [2, 1, 2]]])
        self.assertEqual(cur_candidates_num, 1)
        self.assertEqual(num_candidates_per_block, [1, 1, 1])
        self.assertEqual(min_candidate_error, 0)

    def test_single_block_error(self):
        bob = Bob(MbCfg(q=3, num_blocks=3, block_length=3, radius=1), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([0], np.array([[[0, 1], [1, 1], [2, 0]]]), np.array([2, 0]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0]], [[0, 0, 1]]])
        self.assertEqual(cur_candidates_num, 2)
        self.assertEqual(num_candidates_per_block, [2])
        self.assertEqual(min_candidate_error, 0)

    def test_multi_block_error(self):  # This fails but we're fine with this because the significance of radius has changed (single-block radius instead of multi-block radius)
        bob = Bob(MbCfg(q=3, num_blocks=3, block_length=3, radius=1), [0, 1, 2, 2, 1, 2, 0, 0, 1])

        bob.decode_multi_block([0], np.array([[[0, 1], [1, 1], [2, 0]]]), np.array([2, 0]))
        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([1], np.array([[[2, 1], [1, 1], [0, 0]]]), np.array([0, 0]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0], [0, 0, 0]], [[1, 2, 0], [0, 0, 1]], [[1, 2, 0], [0, 0, 2]], [[0, 0, 1], [0, 0, 0]], [[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 2]]])
        self.assertEqual(cur_candidates_num, 6)
        self.assertEqual(num_candidates_per_block, [2, 3])
        self.assertEqual(min_candidate_error, 0)

        [cur_candidates_num, num_candidates_per_block, min_candidate_error] = bob.decode_multi_block([1, 2], np.array([[[2, 0], [0, 1], [2, 1]], [[0, 1], [2, 1], [0, 2]]]), np.array([2, 1]))
        self.assertEqual(bob.a_candidates, [[[1, 2, 0], [0, 0, 0], [2, 1, 2]]])
        self.assertEqual(cur_candidates_num, 1)
        self.assertEqual(num_candidates_per_block, [1, 1, 1])
        self.assertEqual(min_candidate_error, 0)

if __name__ == '__main__':
    unittest.main()
