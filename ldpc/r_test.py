import unittest

import numpy as np
from scipy.sparse import csr_matrix

from ldpc.f import F
from ldpc_matrix import LdpcMatrix
from q import Q
from r import R


class RTest(unittest.TestCase):
    def test_init(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 2, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        a = [2, 1, 0]
        encoded_a = encoding_matrix * a  # [2, 0, 1]

        r = R(encoding_matrix, encoded_a)
        np.testing.assert_array_equal(r.toarray(), None)

    def test_update(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 2, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        q = Q(F(3, 0.01, [1, 2, 0]), encoding_matrix)
        a = [2, 1, 0]
        encoded_a = encoding_matrix * a  # [2, 0, 1]

        r = R(encoding_matrix, encoded_a)
        r.update(q)
        np.testing.assert_array_equal(r.toarray(), [[0.495, 0.495, 0.01],
                                                    [0.495, 0.495, 0.01],
                                                    [1.0, 0.0, 0.0],
                                                    [0.49015, 0.254925, 0.254925],
                                                    [0.49015, 0.254925, 0.254925],
                                                    [0.254925, 0.254925, 0.49015]])

    def test_calculate_new_distribution(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 2, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        r = R(encoding_matrix, None,
              [[0.2, 0.7, 0.1], [1.0, 0.0, 0.0], [0.4, 0.3, 0.3], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.6, 0.3, 0.1]])
        f = F(3, 0.01, None, [[0.2, 0.3, 0.5], [0.0, 0.9, 0.1], [0.3, 0.4, 0.3]])

        new_dist = r.calculate_new_distribution(f)
        np.testing.assert_array_equal(new_dist, [[0.0, 0.105, 0.025], [0.0, 0.0, 0.1], [0.072, 0.0, 0.0]])


if __name__ == '__main__':
    unittest.main()
