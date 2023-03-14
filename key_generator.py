import numpy as np


class KeyGenerator(object):

    def __init__(self, p_err, N, base=3):
        self.m = base
        self.p_err = float(p_err)
        self.N = N

    def generate_keys(self):
        a = np.random.choice(self.m, self.N)
        b = np.mod(
            np.add(a, np.random.choice(range(self.m), self.N,
                                       p=[self.p_err] + [(1 - self.p_err) / (self.m - 1)] * (self.m - 1))),
            [self.m] * self.N)

        return a, b

    def generate_complement_key(self, x):
        delta = np.add(x, np.random.choice(range(self.m), self.N,
                                           p=[self.p_err] + [(1 - self.p_err) / (self.m - 1)] * (self.m - 1)))
        y = np.mod(delta, [self.m] * self.N)
        return y
