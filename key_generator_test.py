import unittest

from key_generator import KeyGenerator


class KeyGeneratorTest(unittest.TestCase):
    def test_zero_error(self):
        p_err = 0
        N = 10
        keygen = KeyGenerator(p_err, N)
        for i in range(0, 1000):
            a, b = keygen.generate_keys()
            self.assertTrue(set(a).issubset({0, 1, 2}))
            self.assertTrue(set(b).issubset({0, 1, 2}))
            self.assertTrue(all([a_i != b_i for a_i, b_i in zip(a, b)]))

    def test_non_zero_error(self):
        p_err = 0.1
        N = 10
        keygen = KeyGenerator(p_err, N)
        num_errors = 0
        for i in range(0, 1000):
            a, b = keygen.generate_keys()
            self.assertTrue(set(a).issubset({0, 1, 2}))
            self.assertTrue(set(b).issubset({0, 1, 2}))
            num_errors = num_errors + len([i for i in filter(lambda z: z[0] == z[1], zip(a, b))])
        self.assertAlmostEqual(num_errors, 1000, delta=100)


if __name__ == '__main__':
    unittest.main()
