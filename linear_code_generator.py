import numpy as np

import util
from protocol_configs import LinearCodeFormat


class LinearCodeGenerator(object):

    def __init__(self, protocol_configs):
        self.cfg = protocol_configs

    def generate_encoding_matrix(self, num_encoding_columns):
        """
        Generate random encoding matrix M of size n_m*num_encoding_columns with max rank.
        """
        if num_encoding_columns > self.cfg.block_length:
            raise RuntimeError("Can't generate matrix, max rank reached.")

        encoding_matrix = np.random.choice(range(0 if self.cfg.use_zeroes_in_encoding_matrix else 1, self.cfg.base),
                                           (self.cfg.block_length, num_encoding_columns))
        while util.calculate_rank(encoding_matrix, self.cfg.base) < num_encoding_columns:
            encoding_matrix = np.random.choice(range(0 if self.cfg.use_zeroes_in_encoding_matrix else 1, self.cfg.base),
                                               (self.cfg.block_length, num_encoding_columns))
        return encoding_matrix


class MatrixFormat(object):
    def __init__(self, encoding_matrix, encoded_vector):
        self.format = LinearCodeFormat.MATRIX
        self.encoding_matrix = encoding_matrix
        self.encoded_vector = encoded_vector

    def __str__(self):
        return "format: " + str(self.format) + "\nencoding_matrix: " + str(
            self.encoding_matrix) + "\nencoded_vector: " + str(self.encoded_vector)


class AffineSubspaceFormat(object):
    def __init__(self, prefix_encoding_matrix, image_base_source, kernel_base, s0):
        self.format = LinearCodeFormat.AFFINE_SUBSPACE
        self.prefix_encoding_matrix = prefix_encoding_matrix
        self.image_base_source = image_base_source
        self.kernel_base = kernel_base
        self.s0 = s0

    def __str__(self):
        return "format: " + str(self.format) + "\nprefix_encoding_matrix: " + str(
            self.prefix_encoding_matrix) + "\nimage_base_source: " + str(
            self.image_base_source) + "\nkernel_base: " + str(self.kernel_base) + "\ns0: " + str(self.s0)
