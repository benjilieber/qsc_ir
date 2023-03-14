import unittest

from ldpc_generator import LdpcGenerator


class LdpcGeneratorTest(unittest.TestCase):
    def test_generate_ldpc_code1(self):
        n = 35
        num_encoding_columns = 19
        sparsity = 35
        cfg = cfg(base=3, block_length=n, num_blocks=1, sparsity=sparsity)
        code_gen = LdpcGenerator(cfg)
        encoding_matrix = code_gen.generate_gallagher_matrix(num_encoding_columns)
        # print(encoding_matrix.toarray())
        print(encoding_matrix.toarray().shape)

    def test_generate_ldpc_code2(self):
        n = 35
        num_encoding_columns = 19
        sparsity = 4
        cfg = cfg(base=3, block_length=n, num_blocks=1, sparsity=sparsity)
        code_gen = LdpcGenerator(cfg)
        encoding_matrix = code_gen.generate_gallagher_matrix(num_encoding_columns)
        print(encoding_matrix.toarray())
        print(encoding_matrix.toarray().shape)

    def test_generate_random_ldpc_code(self):
        n = 35
        num_encoding_columns = 19
        sparsity = 4
        cfg = cfg(base=3, block_length=n, num_blocks=1, sparsity=sparsity)
        code_gen = LdpcGenerator(cfg)
        encoding_matrix = code_gen.generate_rand_matrix(num_encoding_columns)
        print(encoding_matrix.toarray())
        print(encoding_matrix.toarray().shape)


if __name__ == '__main__':
    unittest.main()
