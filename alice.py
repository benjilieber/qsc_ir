import numpy as np
from linear_code_generator import LinearCodeGenerator
from encoder import Encoder


class Alice(object):

    def __init__(self, protocol_configs, a):
        self.cfg = protocol_configs
        self.linear_code_generator = LinearCodeGenerator(protocol_configs)
        self.encoder = Encoder(protocol_configs.base, protocol_configs.block_length)
        self.a = np.array_split(a, protocol_configs.num_blocks)  # Alice's private key, partitioned into blocks
        self.a_base_m = np.array(list(map(self.encoder.base_3_to_m, self.a)))  # A in base m

    def encode_blocks(self, blocks_indices, number_of_encodings):
        """
        Alice encodes A.
        Picks a nonzero random encoding matrix M of size n_m*k*m.
        Encode A using M, A'.
        Return M and A'.
        """
        encoding_matrices_delta = np.zeros([len(blocks_indices), self.cfg.block_length_base_m, number_of_encodings], dtype=int)
        for i, block_index in enumerate(blocks_indices):
            cur_encoding_matrix_delta = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
            encoding_matrices_delta[i] = cur_encoding_matrix_delta

        # Alice returns M and A'.
        encoded_a = self.encoder.encode_multi_block(encoding_matrices_delta,
                                                    np.take(self.a, blocks_indices, axis=0))
        return encoding_matrices_delta, encoded_a