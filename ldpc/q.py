import copy

import numpy as np


class Q(object):
    def __init__(self, f, encoding_matrix, data=None):
        self.f = f
        self.encoding_matrix = encoding_matrix
        self.base = encoding_matrix.q
        self.indices = encoding_matrix.indices
        self.indptr = encoding_matrix.indptr
        self.indices_r = encoding_matrix.indices_r
        self.indptr_r = encoding_matrix.indptr_r
        if data is not None:
            self.data = data
        else:
            self.data = np.array([f[j] for j in encoding_matrix.indices])

    def update(self, r):
        """
        Symbol probabilities: q[i][j][s] is the probability that a_j == s given the information obtained via
        checks I[j]\{i}.
        """
        self.data = np.array(
            [normalize([self.f[j, s] * np.prod(
                [r.data[k][s] for k in self.indices_r[self.indptr_r[j]:self.indptr_r[j + 1]] if k != i + j_ord]) for s
                        in
                        range(self.base)])
             for i_ord, i in enumerate(self.indptr[:-1])
             for j_ord, j in enumerate(self.indices[self.indptr[i_ord]:self.indptr[i_ord + 1]])])

    def fork(self, forked_index, forked_values, forked_f_list=None):
        forked_q_list = []
        for s in forked_values:
            forked_q_data = copy.deepcopy(self.data)
            forked_index_dist = [0.0 + 1.0 * (s == s_i) for s_i in range(self.base)]
            for ind in self.indices_r[self.indptr_r[forked_index]:self.indptr_r[forked_index + 1]]:
                forked_q_data[ind] = forked_index_dist
            f = forked_f_list[s] if forked_f_list else self.f
            forked_q_list.append(Q(f, self.encoding_matrix, forked_q_data))
        return forked_q_list

    def __getitem__(self, key):
        return self.data[key]

    def toarray(self):
        return self.data


def normalize(raw_list):
    return [i / sum(raw_list) for i in raw_list]
