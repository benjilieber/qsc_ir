import os
import sys
import math
import time

import util
from itertools import product
import itertools

from ldpc.ldpc_matrix import LdpcMatrix

sys.path.append(os.getcwd())

import unittest
from ldpc_decoder import LdpcDecoder
import numpy as np
from ldpc_generator import LdpcGenerator
from protocol_configs import ProtocolConfigs
from protocol_configs import CodeGenerationStrategy
from encoder import Encoder
from key_generator import KeyGenerator
from scipy.sparse import csr_matrix


class LdpcDecoderTest(unittest.TestCase):
    def test_decode_easy(self): # successful
        indptr = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        indices = [1, 3, 0, 1, 1, 2, 0, 3, 0, 3, 1, 2, 2, 3, 0, 2]
        data = [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1]
        encoding_matrix = LdpcMatrix(csr_matrix((data, indices, indptr), shape=(8, 4)), num_noise_symbols=4, base=3)
        # encoding_matrix = [[0, 1, 0, 1, 1, 0, 0, 2],
        #                    [2, 1, 1, 0, 0, 1, 0, 0],
        #                    [0, 0, 1, 0, 0, 2, 1, 1],
        #                    [1, 0, 0, 2, 1, 0, 1, 0]]
        encoded_a = [2, 1, 1, 2, 0, 2, 0, 0]
        b = [0, 2, 2, 0]
        decoder = LdpcDecoder(3, 0.01, encoding_matrix, encoded_a)
        _, stats = decoder.decode_belief_propagation(b, 20)
        a_guess = stats.a_guess_list[0]
        np.testing.assert_array_equal(encoding_matrix* a_guess, encoded_a)

    def test_decode_square_matrix_p0(self):
        n = 10000
        p_err = 0.0
        m = 10000
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p0(self):
        n = 900
        p_err = 0.0
        m = 800
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p0(self):
        n = 900
        p_err = 0.0
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_square_matrix_p001(self):
        n = 10000
        p_err = 0.01
        m = 10000
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p001(self):
        n = 900
        p_err = 0.01
        m = 800
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p001(self):
        n = 900
        p_err = 0.01
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_square_matrix_p002(self):
        n = 10000
        p_err = 0.02
        m = 10000
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p002(self):
        n = 900
        p_err = 0.02
        m = 800
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p002(self):
        n = 10000
        p_err = 0.02
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test(n, m, p_err, sparsity)

def run_single_test(n, m, p_err, sparsity):
    cfg = ProtocolConfigs(base=3, block_length=n, num_blocks=1,
                          code_generation_strategy=CodeGenerationStrategy.LDPC_CODE, sparsity=sparsity)
    code_gen = LdpcGenerator(cfg)
    np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    encoding_matrix = code_gen.generate_gallagher_matrix(m)
    key_generator = KeyGenerator(p_err=p_err, key_length=n)
    a, b = key_generator.generate_keys()
    encoded_a = encoding_matrix * a

    decoder = LdpcDecoder(3, p_err, encoding_matrix, encoded_a)
    _, stats = decoder.decode_belief_propagation(b, 20)
    a_guess = stats.a_guess_list[0]
    np.testing.assert_array_equal(a_guess, a)

if __name__ == '__main__':
    unittest.main()
