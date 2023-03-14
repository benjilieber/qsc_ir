import collections
import random
from enum import Enum

import numpy as np

from ldpc.profile import Profile
from ldpc.profile import ProfileType


class ConstructorMethod(Enum):
    POISSON = 1
    PERMUTATIONS = 2


class Constructor(object):
    def __init__(self, constructor_method, profile):
        self.constructor_method = constructor_method
        self.profile = profile
        self.num_noise_symbols = profile.num_noise_symbols
        self.num_checks = profile.num_checks

    def construct(self):
        if self.constructor_method == ConstructorMethod.POISSON:
            return self.construct_poisson()
        elif self.constructor_method == ConstructorMethod.PERMUTATIONS:
            return self.construct_permutations()

    def construct_poisson(self):
        if True:
            col_weights = [w for (w, k) in self.profile.col_profile.items() for _ in range(k)]
            random.shuffle(col_weights)
            row_weights = [w for (w, k) in self.profile.row_profile.items() for _ in range(k)]
            random.shuffle(row_weights)
            col_with_duplicity = [i for i, w in enumerate(col_weights) for _ in range(w)]
            row_with_duplicity = [i for i, w in enumerate(row_weights) for _ in range(w)]
            random.shuffle(row_with_duplicity)
            edges = list(zip(col_with_duplicity, row_with_duplicity))
            print([item for item, count in collections.Counter(edges).items() if count > 1])  # duplicate edges
            cum_col_weights = np.cumsum(col_weights)
            neighbors_per_col = [set(row_with_duplicity[cur_j:next_j]) for cur_j, next_j in
                                 zip(cum_col_weights, cum_col_weights[1:])]
            pair_weights = [len(a.intersection(b)) for idx, a in enumerate(neighbors_per_col) for b in
                            neighbors_per_col[idx + 1:]]
            num_4_cycles = sum(w == 2 for w in pair_weights)
            print(num_4_cycles)

    def construct_permutations(self):
        pass


c = Constructor(ConstructorMethod.POISSON, Profile(12000, 6000, ProfileType.p_93A))
c.construct()
