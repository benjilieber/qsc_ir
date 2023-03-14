from enum import Enum


class ProfileType(Enum):
    p_3 = 1
    p_33 = 2
    p_93P = 3
    p_93A = 4
    p_93X = 5
    p_93Y = 6


class Profile(object):
    def __init__(self, num_noise_symbols, num_checks, profile_type):
        self.num_noise_symbols = num_noise_symbols
        self.num_checks = num_checks
        self.profile_type = profile_type
        assert (self._check_validity())
        self.col_profile = self._create_col_profile()
        self.row_profile = self._create_row_profile()

    def _check_validity(self):
        return {
            ProfileType.p_3: self.num_noise_symbols == 2 * self.num_checks,
            ProfileType.p_33: self.num_noise_symbols == 2 * self.num_checks,
            ProfileType.p_93P: not (self.num_noise_symbols % 12) and (self.num_noise_symbols == 2 * self.num_checks),
            ProfileType.p_93A: not (self.num_noise_symbols % 12) and (self.num_noise_symbols == 2 * self.num_checks),
            ProfileType.p_93X: not (self.num_noise_symbols % 12) and (self.num_noise_symbols == 2 * self.num_checks),
            ProfileType.p_93Y: not (self.num_noise_symbols % 12) and (self.num_noise_symbols == 2 * self.num_checks),
        }[self.profile_type]

    def _create_col_profile(self):
        return {
            ProfileType.p_3: {3: self.num_noise_symbols},
            ProfileType.p_33: {3: self.num_noise_symbols},
            ProfileType.p_93P: {3: int(self.num_noise_symbols * 11 / 12), 9: int(self.num_noise_symbols / 12)},
            ProfileType.p_93A: {3: int(self.num_noise_symbols * 11 / 12), 9: int(self.num_noise_symbols / 12)},
            ProfileType.p_93X: {3: int(self.num_noise_symbols * 11 / 12), 9: int(self.num_noise_symbols / 12)},
            ProfileType.p_93Y: {3: int(self.num_noise_symbols * 11 / 12), 9: int(self.num_noise_symbols / 12)},
        }[self.profile_type]

    def _create_row_profile(self):
        return {
            ProfileType.p_3: {6: self.num_checks},
            ProfileType.p_33: {6: self.num_checks},
            ProfileType.p_93P: {7: self.num_checks},
            ProfileType.p_93A: {7: self.num_checks},
            ProfileType.p_93X: {7: self.num_checks},
            ProfileType.p_93Y: {7: self.num_checks},
        }[self.profile_type]
