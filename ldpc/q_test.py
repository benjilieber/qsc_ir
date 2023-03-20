import unittest

import numpy as np
from scipy.sparse import csr_matrix

from ldpc.f import F
from ldpc.q import Q
from ldpc.r import R
from ldpc_matrix import LdpcMatrix


class QTest(unittest.TestCase):
    def test_init(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 2, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)

        q = Q(F(3, 0.01, [1, 2, 0]), encoding_matrix)
        np.testing.assert_array_equal(q.toarray(), [[0.495, 0.01, 0.495], [0.01, 0.495, 0.495], [0.01, 0.495, 0.495],
                                                    [0.495, 0.01, 0.495], [0.495, 0.495, 0.01], [0.01, 0.495, 0.495]])

    def test_update(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 0, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        encoded_a = [2, 0, 1]
        r = R(encoding_matrix, encoded_a,
              [[0.0, 0.25, 0.75], [0.3, 0.4, 0.3], [0.495, 0.495, 0.01], [0.495, 0.01, 0.495],
               [0.25, 0.25, 0.5], [0.0, 0.0, 1.0]])

        q = Q(F(3, 0.01, [1, 2, 0]), encoding_matrix)
        q.update(r)
        np.testing.assert_array_equal(q.toarray(),
                                      [[0.49989799041109867, 0.00020401917780271347, 0.49989799041109867],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 0.006688963210702342, 0.9933110367892977],
                                       [0.495, 0.495, 0.01],
                                       [0.014705882352941176, 0.9705882352941176, 0.014705882352941176]])

    def test_fork(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 0, 1, 2, 1])
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)

        q = Q(F(3, 0.01, [1, 2, 0]), encoding_matrix)
        q_list = q.fork(2, [0, 1, 2])
        np.testing.assert_array_equal(q_list[0].toarray(), [[0.495, 0.01, 0.495], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                                            [0.495, 0.01, 0.495], [0.495, 0.495, 0.01],
                                                            [1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(q_list[1].toarray(), [[0.495, 0.01, 0.495], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                                                            [0.495, 0.01, 0.495], [0.495, 0.495, 0.01],
                                                            [0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(q_list[2].toarray(), [[0.495, 0.01, 0.495], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                                            [0.495, 0.01, 0.495], [0.495, 0.495, 0.01],
                                                            [0.0, 0.0, 1.0]])


if __name__ == '__main__':
    unittest.main()
