import numpy as np
import random
import math
from scipy.sparse import csr_matrix
import scipy
from ldpc_matrix import LdpcMatrix

class LdpcGenerator(object):
    def __init__(self, protocol_configs):
        self.cfg = protocol_configs

    def generate_gallagher_matrix(self, num_encoding_columns):
        """
        Generate random LDPC sparse matrix M of size n_m*num_encoding_columns with max rank.
        According to Gallagher algorithm.
        """
        n = self.cfg.block_length
        s = self.cfg.sparsity
        t = max(math.floor(s * num_encoding_columns / n), 1)

        num_encodings_list = partition_almost_uniformly(num_encoding_columns, t)
        sub_matrices = [self.__generate_gallagher_submatrix(n, num_encodings_list[i]) for i in range(t)
                        if num_encodings_list[i] != 0]
        return LdpcMatrix(scipy.sparse.vstack(sub_matrices), n, self.cfg.q)

    def __generate_gallagher_submatrix(self, n, num_subgraph_encodings):
        data = np.array([random.randint(1, self.cfg.q - 1) for _ in range(n)])

        permutation = np.random.permutation(n)
        indices = permutation

        partition = partition_almost_uniformly(n, num_subgraph_encodings)
        cum_partition = np.cumsum(partition)
        indptr = np.insert(cum_partition, 0, 0)

        return csr_matrix((data, indices, indptr), shape=(num_subgraph_encodings, n))

    def generate_rand_matrix(self, num_encoding_columns):
        n = self.cfg.block_length
        p = self.cfg.sparsity / n

        check_degs = [max(1, raw_deg) for raw_deg in np.random.binomial(n, p, num_encoding_columns)]
        indptr = np.insert(np.cumsum(check_degs), 0, 0)
        indices = np.array([ind for deg in check_degs for ind in sorted(np.random.choice(range(0, n), deg, replace=False))])
        # TODO: ensure each column has atleast one 1/2

        data = np.array([random.randint(1, self.cfg.q - 1) for _ in range(indptr[-1])])

        return LdpcMatrix(csr_matrix((data, indices, indptr), shape=(num_encoding_columns, n)), n, self.cfg.q)





def partition_almost_uniformly(num_elements, buckets):
    """" Determine num encodings per subgraph """
    spillover = num_elements % buckets
    num_elements_list = [num_elements // buckets + 1] * spillover + [num_elements // buckets] * (buckets - spillover)
    random.shuffle(num_elements_list)
    return num_elements_list