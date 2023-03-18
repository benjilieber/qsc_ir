import copy
import math
import numpy as np


class F(object):
    def __init__(self, base, p_err, b=None, data=None, use_log=False):
        self.base = base
        self.p_err = p_err
        if data is not None:
            self.data = np.array(data)
        else:
            self.data = np.array([[self._calculate_f_helper(s, b_i) for s in range(base)] for b_i in b])
            if use_log:
                self.data = np.array(
                    [[-math.inf if (val == 0.0) else math.log(val) for val in row] for row in self.data])

    def _calculate_f_helper(self, s, b_i):
        f_raw = self.p_err if b_i == s else (1 - self.p_err) / (self.base - 1)
        return f_raw

    def fork(self, forked_index, forked_values):
        forked_f_list = []
        for s in forked_values:
            forked_f_data = copy.deepcopy(self.data)
            forked_index_dist = [0.0 + 1.0 * (s == s_i) for s_i in range(self.base)]
            forked_f_data[forked_index] = forked_index_dist
            forked_f_list.append(F(self.p_err, self.base, None, forked_f_data))
        return forked_f_list

    def __getitem__(self, key):
        return self.data[key]
