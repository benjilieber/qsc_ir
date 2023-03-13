import unittest

import numpy as np

from encoder import Encoder


class EncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder(3, 3)

    def test_encode_single_block(self):
        np.testing.assert_array_equal(self.encoder.encode_single_block([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 0]),
                                      np.asarray([1, 2, 0]))
        np.testing.assert_array_equal(self.encoder.encode_single_block([[1, 0], [0, 1], [0, 0]], [1, 2, 0]),
                                      np.asarray([1, 2]))
        np.testing.assert_array_equal(self.encoder.encode_single_block([[2, 0, 1], [1, 1, 0], [2, 1, 2]], [0, 2, 1]),
                                      np.asarray([1, 0, 2]))
        np.testing.assert_array_equal(self.encoder.encode_single_block([[2, 0], [1, 1], [2, 1]], [0, 2, 1]),
                                      np.asarray([1, 0]))

    def test_encode_multi_block(self):
        np.testing.assert_array_equal(self.encoder.encode_multi_block([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], [[1, 2, 0]]),
                                      np.asarray([1, 2, 0]))
        np.testing.assert_array_equal(
            self.encoder.encode_multi_block([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                                            [[1, 2, 0], [2, 2, 1]]),
            np.asarray([0, 1, 1]))
        np.testing.assert_array_equal(
            self.encoder.encode_multi_block([[[1, 0], [0, 1], [0, 0]], [[1, 0], [0, 1], [0, 0]]],
                                            [[1, 2, 0], [2, 2, 1]]),
            np.asarray([0, 1]))
        np.testing.assert_array_equal(
            self.encoder.encode_multi_block([[[2, 0, 1], [1, 1, 0], [2, 1, 2]], [[0, 0, 1], [2, 1, 0], [1, 2, 2]]],
                                            [[0, 2, 1], [2, 0, 1]]),
            np.asarray([2, 2, 0]))
        np.testing.assert_array_equal(
            self.encoder.encode_multi_block([[[2, 0], [1, 1], [2, 1]], [[0, 0], [2, 1], [1, 2]]],
                                            [[0, 2, 1], [2, 0, 1]]),
            np.asarray([2, 2]))

    def test_get_missing_encoding_delta(self):
        self.assertListEqual(self.encoder.get_missing_encoding_delta([1, 2, 0], [0, 0, 1]), [1, 2, 2])
        self.assertListEqual(self.encoder.get_missing_encoding_delta([2, 2, 0, 1, 0], [0, 1, 2, 0, 2]), [2, 1, 1, 1, 1])

if __name__ == '__main__':
    unittest.main()
