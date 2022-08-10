import numpy as np

import util


class LinearCodeGenerator(object):

    def __init__(self, protocol_configs):
        self.cfg = protocol_configs

    def generate_encoding_matrix(self, num_encoding_columns):
        """
        Generate random encoding matrix M of size n_m*num_encoding_columns with max rank.
        """
        if num_encoding_columns > self.cfg.block_length_base_m:
            raise RuntimeError("Can't generate matrix, max rank reached.")

        encoding_matrix = np.random.choice(range(0 if self.cfg.use_zeroes_in_encoding_matrix else 1, self.cfg.base),
                                           (self.cfg.block_length_base_m, num_encoding_columns))
        while util.calculate_rank(encoding_matrix, self.cfg.base) < num_encoding_columns:
            encoding_matrix = np.random.choice(range(0 if self.cfg.use_zeroes_in_encoding_matrix else 1, self.cfg.base),
                                               (self.cfg.block_length_base_m, num_encoding_columns))
        return encoding_matrix