import math
import unittest

import numpy as np

from key_generator import KeyGenerator
from ldpc.f import F
from ldpc.ldpc_generator import LdpcGenerator
from ldpc.q import Q
from ldpc.r import R


class Tests(unittest.TestCase):
    def test_f_forked_influence(self):
        n = 35
        num_encoding_columns = 19
        sparsity = 3
        cfg = cfg(base=3, block_length=n, num_blocks=1, sparsity=sparsity)
        code_gen = LdpcGenerator(cfg)
        encoding_matrix = code_gen.generate_gallagher_matrix(num_encoding_columns)
        key_generator = KeyGenerator(p_err=0.01, key_length=n, base=3)
        a, b = key_generator.generate_keys()
        encoded_a = encoding_matrix * a
        forked_index = 6
        forked_values = list(range(3))
        original_f = F(3, 0.01, b)
        forked_f = original_f.fork(forked_index, forked_values)
        original_q = Q(original_f, encoding_matrix)
        forked_q = original_q.fork(forked_index, forked_values)
        r = R(encoding_matrix, encoded_a)
        r.update(original_q)

        for i in range(3):
            r.update(forked_q[i])
            forked_q[i].update(r)
            qdata1 = forked_q[i].data
            forked_q[i].f = forked_f[i]
            forked_q[i].update(r)
            qdata2 = forked_q[i].data
            diff_indices = np.argwhere(qdata1 != qdata2)
            print(diff_indices)
            print(qdata1[diff_indices])
            print(qdata2[diff_indices])
            np.testing.assert_array_equal(qdata1, qdata2)

        # Conclusion: not forking f doesn't propagate the forking completely

    def test_error_probs(self):
        p_err = 0.01
        for d in range(1, 11):
            print(d)
            print(calculate_all_p(d, p_err))


def calculate_all_p(d, p_err):
    return [calculate_p(d1, d - d1, p_err) for d1 in range(d + 1)]


def calculate_p(d1, d2, p_err):
    return sum([calculate_p_helper1(d1, p_err, k) * calculate_p_helper1(d2, p_err, k) for k in range(3)])


def calculate_p_helper1(di, p_err, k):
    if di == 0:
        return k == 0
    return sum([calculate_p_helper2(di, p_err, k, q0) for q0 in range(di + 1)])


def calculate_p_helper2(di, p_err, k, q0):
    return sum([trinomial(di, q0, q1) for q1 in get_q1_range(di, k, q0)]) * calculate_p_for_di_q0(di, p_err, q0)


def trinomial(n, m1, m2):
    return math.factorial(n) / (math.factorial(m1) * math.factorial(m2) * math.factorial(n - m1 - m2))


def get_q1_range(di, k, q0):
    lower_bound = math.ceil((q0 - di - k) / 3)
    upper_bound = math.floor((di - q0 - k) / 3)
    return [int((di - q0 + k + 3 * r) / 2) for r in range(lower_bound, upper_bound + 1) if ((di + q0 + k + r) % 2 == 0)]


def calculate_p_for_di_q0(di, p_err, q0):
    return ((1 + p_err) / 4) ** (di - q0) * ((1 - p_err) / 2) ** q0


def calculate_all_p(d, p_err):
    return [calculate_p(d1, d - d1, p_err) for d1 in range(d + 1)]


def calculate_p(d1, d2, p_err):
    return sum([calculate_p_helper1(d1, p_err, k) * calculate_p_helper1(d2, p_err, k) for k in range(3)])


def calculate_p_helper1(di, p_err, k):
    if di == 0:
        return k == 0
    return sum([calculate_p_helper2(di, p_err, k, q0) for q0 in range(di + 1)])


def calculate_p_helper2(di, p_err, k, q0):
    return sum([trinomial(di, q0, q1) for q1 in get_q1_range(di, k, q0)]) * calculate_p_for_di_q0(di, p_err, q0)


def trinomial(n, m1, m2):
    return math.factorial(n) / (math.factorial(m1) * math.factorial(m2) * math.factorial(n - m1 - m2))


def get_q1_range(di, k, q0):
    lower_bound = math.ceil((q0 - di - k) / 3)
    upper_bound = math.floor((di - q0 - k) / 3)
    return [int((di - q0 + k + 3 * r) / 2) for r in range(lower_bound, upper_bound + 1) if ((di + q0 + k + r) % 2 == 0)]


def calculate_p_for_di_q0(di, p_err, q0):
    return ((1 + p_err) / 4) ** (di - q0) * ((1 - p_err) / 2) ** q0


if __name__ == '__main__':
    unittest.main()
