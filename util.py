import math

import galois
import numpy as np


def hamming_multi_block(x, y):
    assert (len(x) == len(y))
    return sum([hamming_single_block(x_i, y_i) for x_i, y_i in zip(x, y)])


def hamming_single_block(x, y):
    assert (len(x) == len(y))
    return len([i for i in filter(lambda z: z[0] != z[1], zip(x, y))])


def closeness_multi_block(x, y, should_assert=True):
    if should_assert:
        assert (len(x) == len(y))
    return sum([closeness_single_block(x_i, y_i) for x_i, y_i in zip(x, y)])


def closeness_single_block(x, y):
    assert (len(x) == len(y))
    return len([i for i in filter(lambda z: z[0] == z[1], zip(x, y))])


def required_checks(n, base, p_err, candidates_left=1):
    if float(p_err) == 0.0:
        return math.ceil(n * math.log(base - 1, base) - math.log(candidates_left, 3))
    if float(p_err) == 1.0:
        return 0
    return math.ceil(
        n * (-p_err * math.log(p_err, base) - (1 - p_err) * (
                    math.log(1 - p_err, base) - math.log(base - 1, base))) - math.log(candidates_left, 3))

def theoretic_key_rate(p_err):
    if p_err in [0.0, 1.0]:
        return math.log(3/2, 2)
    return math.log(3, 2) + p_err * math.log(p_err, 2) + (1-p_err) * math.log((1-p_err)/2, 2)

def calculate_rank(matrix, base):
    GF = galois.GF(base)
    matrix = GF(matrix)
    matrix.row_reduce()
    return np.linalg.matrix_rank(matrix)
