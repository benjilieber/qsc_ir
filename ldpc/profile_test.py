import unittest

from ldpc.profile import ProfileType
from ldpc.profile import Profile

class ProfileTest(unittest.TestCase):
    def test_init_3_series(self):
        num_noise_symbols = 30
        num_checks = 15
        p3 = Profile(num_noise_symbols, num_checks, ProfileType.p_3)
        self.assertDictEqual(p3.col_profile, {3 : 30})
        self.assertDictEqual(p3.row_profile, {6 : 15})
        p33 = Profile(num_noise_symbols, num_checks, ProfileType.p_33)
        self.assertDictEqual(p33.col_profile, {3 : 30})
        self.assertDictEqual(p33.row_profile, {6 : 15})

    def test_init_93_series(self):
        num_noise_symbols = 36
        num_checks = 18
        p93p = Profile(num_noise_symbols, num_checks, ProfileType.p_93P)
        self.assertDictEqual(p93p.col_profile, {3 : 33, 9 : 3})
        self.assertDictEqual(p93p.row_profile, {7 : 18})
        p93a = Profile(num_noise_symbols, num_checks, ProfileType.p_93A)
        self.assertDictEqual(p93a.col_profile, {3 : 33, 9 : 3})
        self.assertDictEqual(p93a.row_profile, {7 : 18})


if __name__ == '__main__':
    unittest.main()
