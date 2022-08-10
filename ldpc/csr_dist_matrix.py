import numpy as np
from scipy.sparse import csr_matrix

class CsrDistMatrix(csr_matrix):
    def __init__(self, indices, indptr, shape, base):
        self.data = np.array(shape=(shape + (base,)), dtype='float')
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        # super(csr_matrix, self).__init__(matrix)
        self.base = base

