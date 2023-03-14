import numpy as np


class Encoder(object):

    def __init__(self, base, block_length):
        self.base = base  # basis field size
        self.block_length = block_length  # block length

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
            [self.encode_single_block(encoding_matrix[i], x_i) for i, x_i in enumerate(x)]), axis=0) % self.base

    def encode_single_block(self, encoding_matrix, x):
        """
        Encode x using M, giving x'.
        x' = x * M
        Shape:
        - x: (n,)
        - M: (n, num_encodings)
        - x': (n,)
        """
        return np.matmul(x, encoding_matrix) % self.base

    def is_single_block_solution(self, encoding_matrix, candidate, encoded_a):
        return np.array_equal(self.encode_single_block(encoding_matrix, candidate), encoded_a)

    def is_multi_block_solution(self, encoding_matrix, candidate, encoded_a):
        return np.array_equal(self.encode_multi_block(encoding_matrix, candidate),
                              encoded_a)

    def get_missing_encoding_delta(self, goal_encoding, cur_encoding):
        return [(x - y) % self.base for x, y in zip(goal_encoding, cur_encoding)]
