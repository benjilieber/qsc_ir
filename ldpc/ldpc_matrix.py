import numpy as np
from scipy.sparse import csr_matrix

class LdpcMatrix(csr_matrix):
    def __init__(self, matrix, num_noise_symbols, base):
        super(csr_matrix, self).__init__(matrix)
        self.base = base
        j_to_ind = [[] for _ in range(num_noise_symbols)]
        for cur_i_start, next_i_start in zip(self.indptr, self.indptr[1:]):
            for j_ord, j in enumerate(self.indices[cur_i_start: next_i_start]):
                j_to_ind[j] += [cur_i_start + j_ord]
        self.indptr_r = np.cumsum([0] + [len(ind_list) for ind_list in j_to_ind])
        self.indices_r = np.array([ind for ind_list in j_to_ind for ind in ind_list])

    def __mul__(self, vector):
        return np.mod(super(csr_matrix, self).__mul__(vector), self.base)