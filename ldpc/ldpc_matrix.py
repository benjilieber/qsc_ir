import numpy as np
from scipy.sparse import csr_matrix

class LdpcMatrix(csr_matrix):
    def __init__(self, matrix, num_noise_symbols, base):
        super(csr_matrix, self).__init__(matrix)
        self.base = base
        j_to_ind = [[] for _ in range(num_noise_symbols)]
        self.j_to_i = [[] for _ in range(num_noise_symbols)]
        for i, (cur_i_start, next_i_start) in enumerate(zip(self.indptr, self.indptr[1:])):
            for j_ord, j in enumerate(self.indices[cur_i_start: next_i_start]):
                j_to_ind[j] += [cur_i_start + j_ord]
                self.j_to_i[j].append(i)
        self.indptr_r = np.cumsum([0] + [len(ind_list) for ind_list in j_to_ind])
        self.indices_r = np.array([ind for ind_list in j_to_ind for ind in ind_list])

    def __mul__(self, vector):
        # return np.mod([np.inner(self.data[self.indices[i_start:i_end]], vector[self.indices[i_start:i_end]]) for i, (i_start, i_end) in enumerate(zip(self.indptr, self.indptr[1:]))], self.base)
        return np.mod(super(csr_matrix, self).__mul__(vector), self.base)

    def __getitem__(self, i):
        return np.array(self.data[self.indices[self.indptr[i]:self.indptr[i+1]]])