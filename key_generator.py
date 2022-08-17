import numpy as np


class KeyGenerator(object):

    def __init__(self, p_err, key_length, base=3):
        self.m = base
        self.p_err = float(p_err)
        self.key_length = key_length

    def generate_keys(self):
        a = np.random.choice(self.m, self.key_length)
        b = np.mod(
            np.add(a, np.random.choice(range(self.m), self.key_length,
                                       p=[self.p_err] + [(1 - self.p_err) / (self.m-1)] * (self.m-1))),
            [self.m] * self.key_length)

        return a, b

    def generate_complement_key(self, x):
        delta = np.add(x, np.random.choice(range(self.m), self.key_length,
                                       p=[self.p_err] + [(1 - self.p_err) / (self.m-1)] * (self.m-1)))
        y = np.mod(delta, [self.m] * self.key_length)
        return y
