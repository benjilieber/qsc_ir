import numpy as np


class R(object):
    def __init__(self, encoding_matrix, encoded_a, data=None):
        """
        Check probabilities: r[i][j][s] is the probability that check i is satisfied (a * encoding_matrix[:][i] == encoded_a[i])
        if a_j == s and the other noise symbols have a separable distribution given by the probabilities q[i][J[i]\{j}].
        """
        self.encoding_matrix = encoding_matrix
        self.encoded_a = encoded_a
        self.base = encoding_matrix.base
        self.indices = encoding_matrix.indices
        self.indptr = encoding_matrix.indptr
        self.indices_r = encoding_matrix.indices_r
        self.indptr_r = encoding_matrix.indptr_r
        self.data = data

    def update(self, q):
        sigma = np.full((len(q.data), self.base), 0.0)
        cur_row = 0
        next_row_start = self.indptr[0]
        for cur in range(len(q.data)):
            enc_coef = self.encoding_matrix.data[cur]
            if next_row_start == cur:
                cur_row += 1
                next_row_start = self.indptr[cur_row]
                sigma[cur] = [
                    np.sum([q.data[cur, t] for t in range(self.base) if ((enc_coef * t) % self.base == s)] or [0]) for s in
                    range(self.base)]
            else:
                sigma[cur] = [
                    np.sum([sigma[cur - 1, (s - enc_coef * t) % self.base] * q.data[cur, t] for t in range(self.base)]) for s in
                    range(self.base)]

        rho = np.full((len(q.data), self.base), 0.0)
        cur_row = len(self.indptr) - 1
        next_row_start = self.indptr[cur_row] - 1
        for cur in reversed(range(len(q.data))):
            enc_coef = self.encoding_matrix.data[cur]
            if next_row_start == cur:
                cur_row -= 1
                next_row_start = self.indptr[cur_row] - 1
                rho[cur] = [
                    np.sum([q.data[cur, t] for t in range(self.base) if ((enc_coef * t) % self.base == s)] or [0]) for s in
                    range(self.base)]
            else:
                rho[cur] = [
                    np.sum([rho[cur + 1, (s - enc_coef * t) % self.base] * q.data[cur, t] for t in range(self.base)]) for s in
                    range(self.base)]

        self.data = np.array([[np.sum([(sigma[cur - 1, (self.encoded_a[cur_i] - self.encoding_matrix.data[
            cur] * s - t) % self.base] if cur > cur_i_start else (
                ((self.encoded_a[cur_i] - self.encoding_matrix.data[cur] * s - t) % self.base) == 0)) * (
                                           rho[cur + 1, t] if (cur < next_i_start - 1) else (t == 0)) for t in
                                       range(self.base)]) for s in range(self.base)] for
                              cur_i, (cur_i_start, next_i_start) in enumerate(zip(self.indptr, self.indptr[1:])) for cur
                              in range(cur_i_start, next_i_start)])

    def calculate_new_distribution(self, f):
        return [[f[j, s] * np.prod(
                [self.data[ind][s] for ind in self.indices_r[cur_j_start:next_j_start]]) for s in
                        range(self.base)]
             for j, (cur_j_start, next_j_start) in enumerate(zip(self.indptr_r, self.indptr_r[1:]))]

    def __getitem__(self, key):
        return self.data[key]

    def toarray(self):
        return self.data
