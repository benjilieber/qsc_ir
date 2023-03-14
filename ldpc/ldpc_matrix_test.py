import unittest

import numpy as np
from scipy.sparse import csr_matrix

from ldpc_matrix import LdpcMatrix


class LdpcMatrixTest(unittest.TestCase):
    def test_init(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 0, 1, 2, 1])
        ldpc_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        np.testing.assert_array_equal(ldpc_matrix.indptr_r, [0, 2, 3, 6])
        np.testing.assert_array_equal(ldpc_matrix.indices_r, [0, 3, 4, 1, 2, 5])

    def test_toarray(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 0, 1, 2, 1])
        original_matrix = csr_matrix((data, indices, indptr), shape=(3, 3))
        ldpc_matrix = LdpcMatrix(original_matrix, num_noise_symbols=3, base=3)
        np.testing.assert_array_equal(original_matrix.toarray(), ldpc_matrix.toarray())

    def test_mul(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 0, 1, 2, 1])
        ldpc_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(3, 3)), num_noise_symbols=3, base=3)
        np.testing.assert_array_equal(ldpc_matrix * [1, 1, 1], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
