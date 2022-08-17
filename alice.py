import random

import numpy as np
from linear_code_generator import LinearCodeGenerator, MatrixFormat, AffineSubspaceFormat
from encoder import Encoder
from protocol_configs import LinearCodeFormat


class Alice(object):

    def __init__(self, protocol_configs, a):
        self.cfg = protocol_configs
        self.linear_code_generator = LinearCodeGenerator(protocol_configs)
        self.encoder = Encoder(protocol_configs.base, protocol_configs.block_length)
        self.a = np.array_split(a, protocol_configs.num_blocks)  # Alice's private key, partitioned into blocks
        self.a_base_m = np.array(list(map(self.encoder.hash_base_to_base, self.a)))  # A in base m

    def encode_blocks(self, blocks_indices, number_of_encodings, format):
        """
        Alice encodes A.
        Picks a nonzero random encoding matrix M of size n_m*k*m.
        Encode A using M, A'.
        Return M and A'.
        """
        if format == LinearCodeFormat.MATRIX:
            encoding_matrices_delta = np.zeros([len(blocks_indices), self.cfg.block_length_hash_base, number_of_encodings], dtype=int)
            for i, block_index in enumerate(blocks_indices):
                cur_encoding_matrix_delta = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
                encoding_matrices_delta[i] = cur_encoding_matrix_delta

            # Alice returns M and A'.
            encoded_a = self.encoder.encode_multi_block(encoding_matrices_delta,
                                                        np.take(self.a, blocks_indices, axis=0))
            return MatrixFormat(encoding_matrices_delta, encoded_a)

        elif format == LinearCodeFormat.AFFINE_SUBSPACE:
            # Encoding matrix for the prefix
            if len(blocks_indices) > 1:
                prefix_encoding_matrices = np.zeros([len(blocks_indices) - 1, self.cfg.block_length_hash_base, number_of_encodings],
                                                    dtype=int)
                for i, block_index in enumerate(blocks_indices[:-1]):
                    cur_encoding_matrix_delta = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
                    prefix_encoding_matrices[i] = cur_encoding_matrix_delta
                prefix_encoded_a = self.encoder.encode_multi_block(prefix_encoding_matrices,
                                                            np.take(self.a, blocks_indices[:-1], axis=0))
            else:
                prefix_encoding_matrices = np.array([])
                prefix_encoded_a = np.zeros(number_of_encodings, dtype=int)

            # Random bases for image and kernel for last block.
            last_block_matrix = self.linear_code_generator.generate_encoding_matrix(self.cfg.block_length)
            image_base_source = last_block_matrix[:number_of_encodings]
            kernel_base = last_block_matrix[number_of_encodings:]
            prefix_encoded_a_source = self.encoder.encode_single_block(image_base_source, prefix_encoded_a)
            random_kernel_element = self.encoder.encode_single_block(kernel_base, np.random.choice(range(self.cfg.base), len(kernel_base)))
            s0 = (prefix_encoded_a_source + self.a[blocks_indices[-1]] + random_kernel_element) % self.cfg.base

            return AffineSubspaceFormat(prefix_encoding_matrices, image_base_source, kernel_base, s0)

        else:
            pass