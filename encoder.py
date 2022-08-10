import math
import numpy as np


class Encoder(object):

    def __init__(self, m, block_length):
        self.m = m  # basis field size
        self.block_length = block_length  # block length
        self.block_length_base_m = math.ceil(block_length * math.log(3, m))  # l in base m

    def encode_multi_block(self, encoding_matrix, x):
        """
        Encode x using M, giving x'.
        x' = sum (x_i * M_i)
        Shape:
        - x: (num_blocks, 1, n)
        - M: (num_blocks, n, num_encodings)

        - x': (n,)
        """
        return np.sum(np.array(
            [self.encode_single_block(encoding_matrix[i], x_i) for i, x_i in enumerate(x)]), axis=0) % self.m

    def encode_single_block(self, encoding_matrix, x):
        """
        Encode x using M, giving x'.
        x' = x * M
        Shape:
        - x: (n,)
        - M: (n, num_encodings)
        - x': (n,)
        """
        return np.matmul(self.base_3_to_m(x), encoding_matrix) % self.m

    def base_3_to_m(self, x):
        if self.m == 3:
            return x
        int_x = int(''.join([str(i) for i in x]), base=3)
        x_base_m_str = np.base_repr(int_x, base=self.m)
        return np.array(list(x_base_m_str.zfill(self.block_length_base_m))).astype(dtype=int)

    def is_single_block_solution(self, encoding_matrix, candidate, encoded_a):
        return np.array_equal(self.encode_single_block(encoding_matrix, candidate), encoded_a)

    def is_multi_block_solution(self, encoding_matrix, candidate, encoded_a):
        return np.array_equal(self.encode_multi_block(encoding_matrix, candidate),
                              encoded_a)

    def get_missing_encoding_delta(self, goal_encoding, cur_encoding):
        return [(x - y) % self.m for x, y in zip(goal_encoding, cur_encoding)]
