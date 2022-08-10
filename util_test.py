import math
import unittest

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import numpy as np

import util

class UtilTest(unittest.TestCase):

    def test_hamming_single_block(self):
        self.assertEqual(util.hamming_single_block([0,1], [0,1]), 0)
        self.assertEqual(util.hamming_single_block([0,1], [1,1]), 1)
        self.assertEqual(util.hamming_single_block([0,1], [1,2]), 2)
        with self.assertRaises(AssertionError):
            util.hamming_single_block([0, 1], [1, 2, 3])
        with self.assertRaises(AssertionError):
            util.hamming_single_block([0, 1, 3], [1, 2])

    def test_hamming_multi_block(self):
        self.assertEqual(util.hamming_multi_block([[0, 1], [1, 2]], [[0,1], [1, 2]]), 0)
        self.assertEqual(util.hamming_multi_block([[0, 1], [1, 2]], [[0,2], [1, 2]]), 1)
        self.assertEqual(util.hamming_multi_block([[0, 1], [1, 2]], [[0,2], [0, 2]]), 2)
        self.assertEqual(util.hamming_multi_block([[0, 1], [1, 2]], [[1,2], [0, 2]]), 3)
        self.assertEqual(util.hamming_multi_block([[0, 1], [1, 2]], [[1,2], [0, 1]]), 4)
        with self.assertRaises(AssertionError):
            util.hamming_multi_block([[0, 1], [1, 2]], [[1, 2], [0, 1], [1, 3]])
        with self.assertRaises(AssertionError):
            util.hamming_multi_block([[0, 1], [1, 2]], [[1, 2], [0, 1, 3]])

    def test_closeness_single_block(self):
        self.assertEqual(util.closeness_single_block([0,1], [0,1]), 2)
        self.assertEqual(util.closeness_single_block([0,1], [1,1]), 1)
        self.assertEqual(util.closeness_single_block([0,1], [1,2]), 0)
        with self.assertRaises(AssertionError):
            util.closeness_single_block([0, 1], [1, 2, 3])
        with self.assertRaises(AssertionError):
            util.closeness_single_block([0, 1, 3], [1, 2])

    def test_closeness_multi_block(self):
        self.assertEqual(util.closeness_multi_block([[0, 1], [1, 2]], [[0,1], [1, 2]]), 4)
        self.assertEqual(util.closeness_multi_block([[0, 1], [1, 2]], [[0,2], [1, 2]]), 3)
        self.assertEqual(util.closeness_multi_block([[0, 1], [1, 2]], [[0,2], [0, 2]]), 2)
        self.assertEqual(util.closeness_multi_block([[0, 1], [1, 2]], [[1,2], [0, 2]]), 1)
        self.assertEqual(util.closeness_multi_block([[0, 1], [1, 2]], [[1,2], [0, 1]]), 0)
        with self.assertRaises(AssertionError):
            util.closeness_multi_block([[0, 1], [1, 2]], [[1, 2], [0, 1], [1, 3]])
        with self.assertRaises(AssertionError):
            util.closeness_multi_block([[0, 1], [1, 2]], [[1, 2], [0, 1, 3]])

    def test_required_checks(self):
        self.assertEqual(util.required_checks(200, 3, 0), math.ceil(200 * math.log(2, 3)))
        self.assertEqual(util.required_checks(200, 3, 0.01), math.ceil(200 * (-0.01*math.log(0.01, 3) - 0.99*math.log(0.99/2, 3))))
        self.assertEqual(util.required_checks(200, 3, 1), 0)
        plot(list(np.arange(0, 1, 0.01)), [util.required_checks(100, 3, p_err) for p_err in np.arange(0, 1, 0.01)])
        plt.show()

    def test_calculate_rank(self):
        base = 3
        encoding_matrix = np.identity(3, dtype=int)
        self.assertEqual(util.calculate_rank(encoding_matrix, base), 3)
        encoding_matrix = np.array([[1, 1, 1], [0, 1, 2], [1, 2, 0]])
        self.assertEqual(np.linalg.matrix_rank(encoding_matrix), 3)
        self.assertEqual(util.calculate_rank(encoding_matrix, base), 2)