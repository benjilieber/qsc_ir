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

    def test_decode_square_matrix_p0_bp(self):
        n = 10000
        p_err = 0.0
        m = 10000
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_square_matrix_p0_it1(self):
        n = 1000
        p_err = 0.0
        m = 1000
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=100)

    def test_decode_square_matrix_p0_it2(self):
        n = 200
        p_err = 0.0
        m = 200
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.9)

    def test_decode_close_to_square_matrix_p0_bp(self):
        n = 900
        p_err = 0.0
        m = 800
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p0_it1(self):
        n = 900
        p_err = 0.0
        m = 800
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=400)

    def test_decode_close_to_square_matrix_p0_it2(self):
        n = 900
        p_err = 0.0
        m = 800
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.9)

    def test_decode_hard_matrix_p0_bp(self):
        n = 900
        p_err = 0.0
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p0_it1(self):
        n = 900
        p_err = 0.0
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=400)

    def test_decode_hard_matrix_p0_it2(self):
        n = 50
        p_err = 0.0
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_square_matrix_p001_bp(self):
        n = 10000
        p_err = 0.01
        m = 10000
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_square_matrix_p001_it1(self):
        n = 100
        p_err = 0.01
        m = 100
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=100)

    def test_decode_square_matrix_p001_it2(self):
        n = 100
        p_err = 0.01
        m = 100
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_close_to_square_matrix_p001_bp(self):
        n = 900
        p_err = 0.01
        m = 800
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p001_it1(self):
        n = 450
        p_err = 0.01
        m = 400
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=80)

    def test_decode_close_to_square_matrix_p001_it2(self):
        n = 405
        p_err = 0.01
        m = 360
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_hard_matrix_p001_bp(self):
        n = 900
        p_err = 0.01
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p001_it1(self):
        n = 80
        p_err = 0.01
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=300)

    def test_decode_hard_matrix_p001_it2(self):
        n = 80
        p_err = 0.01
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_square_matrix_p002_bp(self):
        n = 10000
        p_err = 0.02
        m = 10000
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_square_matrix_p002_it1(self):
        n = 80
        p_err = 0.02
        m = 80
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=200)

    def test_decode_square_matrix_p002_it2(self):
        n = 80
        p_err = 0.02
        m = 80
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_close_to_square_matrix_p002_bp(self):
        n = 900
        p_err = 0.02
        m = 800
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_close_to_square_matrix_p002_it1(self):
        n = 90
        p_err = 0.02
        m = 80
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=200)

    def test_decode_close_to_square_matrix_p002_it2(self):
        n = 90
        p_err = 0.02
        m = 80
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)

    def test_decode_hard_matrix_p002_bp(self):
        n = 10000
        p_err = 0.0
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_bp(n, m, p_err, sparsity)

    def test_decode_hard_matrix_p002_it1(self):
        n = 80
        p_err = 0.02
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, max_candidates_num=200)

    def test_decode_hard_matrix_p002_it2(self):
        n = 80
        p_err = 0.02
        m = util.required_checks(n, 3, p_err)
        sparsity = 3
        run_single_test_it(n, m, p_err, sparsity, success_rate=0.5)
        

def run_single_test_bp(n, m, p_err, sparsity):
    cfg = ProtocolConfigs(base=3, block_length=n, num_blocks=1,
                          code_generation_strategy=CodeGenerationStrategy.ldpc, sparsity=sparsity)
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

def run_single_test_it(n, m, p_err, sparsity, max_candidates_num=None, success_rate=None):
    cfg = ProtocolConfigs(base=3, block_length=n, num_blocks=1,
                          code_generation_strategy=CodeGenerationStrategy.ldpc, sparsity=sparsity)
    code_gen = LdpcGenerator(cfg)
    np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    encoding_matrix = code_gen.generate_gallagher_matrix(m)
    key_generator = KeyGenerator(p_err=p_err, key_length=n)
    a, b = key_generator.generate_keys()
    encoded_a = encoding_matrix * a

    decoder = LdpcDecoder(3, p_err, encoding_matrix, encoded_a, a=a, max_candidates_num=max_candidates_num, success_rate=success_rate)
    candidates = decoder.decode_iteratively(b)
    print(candidates)
    print(a)
    print([sum(np.equal(a, c)) for c in candidates])
    print([sum(np.equal(encoded_a, encoding_matrix*c)) for c in candidates])
    print(len(encoded_a))

if __name__ == '__main__':
    unittest.main()
