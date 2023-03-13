import numpy as np
from mb.linear_code_generator import LinearCodeGenerator, MatrixFormat, AffineSubspaceFormat
from mb.encoder import Encoder
from mb.mb_cfg import LinearCodeFormat


class Alice(object):

    def __init__(self, protocol_configs, a):
        self.cfg = protocol_configs
        self.linear_code_generator = LinearCodeGenerator(protocol_configs)
        self.encoder = Encoder(protocol_configs.q, protocol_configs.block_length)
        self.a = np.array_split(a, protocol_configs.num_blocks)  # Alice's private key, partitioned into blocks

    def encode_blocks(self, encode_new_block, blocks_indices, number_of_encodings, encoding_matrix_prefix, encoding_format):
        """
        Alice encodes A.
        Picks a nonzero random encoding matrix M of size n_m*k*m.
        Encode A using M, A'.
        Return M and A'.
        """
        if encoding_format == LinearCodeFormat.MATRIX:
            encoding_matrices_delta = np.zeros([len(blocks_indices), self.cfg.block_length, number_of_encodings], dtype=int)
            encoding_matrices_delta[:len(blocks_indices)-encode_new_block] = encoding_matrix_prefix
            if encode_new_block:
                encoding_matrices_delta[-1] = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)

            # Alice returns M and A'.
            encoded_a = self.encoder.encode_multi_block(encoding_matrices_delta,
                                                        np.take(self.a, blocks_indices, axis=0))
            return MatrixFormat(encoding_matrices_delta, encoded_a)

        elif encoding_format == LinearCodeFormat.AFFINE_SUBSPACE:
            # Encoding matrix for the prefix
            if len(blocks_indices) > 1:
                prefix_encoding_matrices = encoding_matrix_prefix
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
            random_kernel_element = self.encoder.encode_single_block(kernel_base, np.random.choice(range(self.cfg.q), len(kernel_base)))
            s0 = (prefix_encoded_a_source + self.a[blocks_indices[-1]] + random_kernel_element) % self.cfg.q

            return AffineSubspaceFormat(prefix_encoding_matrices, image_base_source, kernel_base, s0)

        else:
            pass