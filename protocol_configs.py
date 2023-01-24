import math
from enum import Enum
from scipy.stats import binom
import signal
import util
import numpy as np


class IndicesToEncodeStrategy(Enum):
    ALL_MULTI_CANDIDATE_BLOCKS = 1
    MOST_CANDIDATE_BLOCKS = 2

class CodeGenerationStrategy(Enum):
    LINEAR_CODE = 'linear'
    LDPC_CODE = 'ldpc'

    def __str__(self):
        return self.value

class RoundingStrategy(Enum):
    FLOOR = 'floor'
    CEIL = 'ceil'

    def __str__(self):
        return self.value

class PruningStrategy(Enum):
    RADII_PROBABILITIES = 'radii_probabilities'
    RELATIVE_WEIGHTS = 'relative_weights'

    def __str__(self):
        return self.value

class LinearCodeFormat(Enum):
    MATRIX = 'matrix'
    AFFINE_SUBSPACE = 'affine_subspace'

    def __str__(self):
        return self.value

class ProtocolConfigs(object):
    """
       m = base field size
       p_err = probability of error in single index
       n = block size
       radius = max number of errors to check during decoding
       goal_candidates_num = number of max candidates we aim to have left at each iteration
       """

    def __init__(self, base, block_length, num_blocks, p_err=0, success_rate=1.0, prefix_radii=None, radius_picking=None, full_rank_encoding=True,
                 use_zeroes_in_encoding_matrix=True,
                 goal_candidates_num=None,
                 max_candidates_num=None,
                 indices_to_encode_strategy=IndicesToEncodeStrategy.ALL_MULTI_CANDIDATE_BLOCKS,
                 rounding_strategy=RoundingStrategy.CEIL,
                 code_generation_strategy=CodeGenerationStrategy.LINEAR_CODE,
                 pruning_strategy=PruningStrategy.RADII_PROBABILITIES,
                 sparsity=None,
                 fixed_number_of_encodings=None,
                 max_num_indices_to_encode=None,
                 encoding_sample_size=1,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        self.verbosity = verbosity
        self.base = base  # basis field size
        self.block_length = block_length  # block length
        self.num_blocks = num_blocks  # number of blocks in private key
        self.key_length = num_blocks * block_length
        self.p_err = p_err
        self.success_rate = success_rate
        self.full_rank_encoding = full_rank_encoding
        self.use_zeroes_in_encoding_matrix = use_zeroes_in_encoding_matrix if base > 2 else True  # Whether to enable zeroes in encoding matrix
        self.goal_candidates_num = goal_candidates_num or block_length ** 2
        self.indices_to_encode_strategy = indices_to_encode_strategy
        self.rounding_strategy = rounding_strategy
        self.code_generation_strategy = code_generation_strategy
        self.pruning_strategy = pruning_strategy
        self.sparsity = sparsity
        self.max_num_indices_to_encode = max_num_indices_to_encode or num_blocks
        self.max_candidates_num = max_candidates_num
        self.encoding_sample_size = encoding_sample_size
        self.theoretic_key_rate = self._theoretic_key_rate()
        self.radius_picking = radius_picking
        if p_err == 0.0:
            self.fixed_radius = True
            self.radius = 0
            self.max_block_error = [0 for _ in range(num_blocks)]
            self.prefix_radii = [0 for _ in range(num_blocks)]
            self.pruning_success_rate = 1.0
        elif radius_picking is False:
            self.fixed_radius = True
            self.radius = self.block_length
            self.pruning_success_rate = success_rate
        else:
            self.fixed_radius = False
            self.max_block_error = self._radius_for_max_block_error()
            self.prefix_radii = prefix_radii or self._determine_prefix_radii()
            self.pruning_success_rate = math.sqrt(success_rate)
            # self.prefix_radii = prefix_radii or self._determine_prefix_radii(timeout)

        self.fixed_number_of_encodings = fixed_number_of_encodings
        if fixed_number_of_encodings:
            self.number_of_encodings_list = self._determine_num_encodings_list()

        self.raw_results_file_path = raw_results_file_path
        self.agg_results_file_path = agg_results_file_path

    def round(self, number):
        if self.rounding_strategy == RoundingStrategy.FLOOR:
            return math.floor(number)
        elif self.rounding_strategy == RoundingStrategy.CEIL:
            return math.ceil(number)
        raise "Bad rounding strategy"

    def _radius_for_max_block_error(self):
        use_product = True

        total_success_prob = (1.0 + self.success_rate) / 2
        # total_success_prob = self.success_rate
        if use_product:
            per_block_success_prob = total_success_prob ** (1 / self.num_blocks)
        else:
            per_block_success_prob = 1 - (1 - total_success_prob) / self.num_blocks

        ceil_k = int(binom.ppf(per_block_success_prob, self.block_length, self.p_err))
        floor_k = ceil_k - 1
        ceil_cdf = binom.cdf(ceil_k, self.block_length, self.p_err)
        floor_cdf = binom.cdf(floor_k, self.block_length, self.p_err)

        if use_product:
            if floor_cdf == 0.0:
                ceil_m = self.num_blocks
            else:
                ceil_m = math.ceil(math.log(total_success_prob, ceil_cdf/floor_cdf) - self.num_blocks * math.log(floor_cdf, ceil_cdf/floor_cdf))
        else:
            ceil_m = math.ceil(self.num_blocks + total_success_prob - 1 - self.num_blocks * floor_cdf)

        floor_m = self.num_blocks - ceil_m
        return [floor_k] * floor_m + [ceil_k] * ceil_m

    def _overall_radius_key_error(self):
        return int(binom.ppf(self.success_rate, self.key_length, self.p_err))

    def _determine_prefix_radii(self):
        radii = np.empty(self.num_blocks, dtype=np.int32)
        success_rate_per_block = 1 - (1 - self.success_rate) / (2 * self.num_blocks)
        # success_rate_per_block = 1 - (1 - self.success_rate) / self.num_blocks

        for i in range(self.num_blocks):
            radii[i] = int(binom.ppf(success_rate_per_block, (i+1) * self.block_length, self.p_err))

        return radii

    def determine_cur_radius(self, last_block_index):
        if self.fixed_radius:
            return self.radius
        return self.max_block_error[last_block_index]

    def _is_within_fixed_radius_single_block(self, x, y):
        return util.closeness_single_block(x, y) <= self.radius

    def _is_within_fixed_radius_multi_block(self, x, y):
        return all([(self._is_within_fixed_radius_single_block(x_i, y_i)) for x_i, y_i in zip(x, y)])

    def _is_within_max_radius_single_block(self, x, y, block_index):
        return util.closeness_single_block(x, y) <= self.max_block_error[block_index]

    def _is_within_max_radius_multi_block(self, x, y):
        return all([(self._is_within_max_radius_single_block(x_i, y_i, i)) for i, (x_i, y_i) in enumerate(zip(x, y))])

    def _is_within_radius_multi_block(self, x, y):
        num_blocks_considered = min(len(x), len(y))
        return util.closeness_multi_block(x[:num_blocks_considered], y[:num_blocks_considered]) <= self.prefix_radii[num_blocks_considered-1]

    def is_within_radius_all_blocks(self, x, y):
        if self.fixed_radius:
            return self._is_within_fixed_radius_multi_block(x, y)
        num_blocks_considered = min(len(x), len(y))
        return all([self._is_within_radius_multi_block(x[:k], y[:k]) for k in range(num_blocks_considered)]) and self._is_within_max_radius_multi_block(x, y)

    def is_within_radius_new_block(self, x, y):
        last_index = min(len(x), len(y)) - 1
        if self.fixed_radius:
            return self._is_within_fixed_radius_single_block(x[last_index], y[last_index])
        return self._is_within_radius_multi_block(x, y) and self._is_within_max_radius_single_block(x[last_index], y[last_index], last_index)

    def _determine_num_encodings_list(self):
        total_checks = util.required_checks(self.key_length, self.base, self.p_err)
        num_encodings_list = []
        tot_num_encodings = 0
        for i in range(self.num_blocks):
            cur_num_encodings = math.floor(total_checks * (i+1) / self.num_blocks - tot_num_encodings)
            num_encodings_list.append(cur_num_encodings)
            tot_num_encodings += cur_num_encodings
        return num_encodings_list
        # Old way:
        # checks_per_block_ceil = math.ceil(total_checks / self.num_blocks)
        # checks_per_block_floor = checks_per_block_ceil - 1
        # m_ceil = total_checks - self.num_blocks * checks_per_block_floor
        # return [checks_per_block_floor] * (self.num_blocks - m_ceil) + [checks_per_block_ceil] * m_ceil

    def _theoretic_key_rate(self):
        if self.p_err == 0.0:
            return math.log(self.base/(self.base-1), 2)
        if self.p_err == 1.0:
            return math.log(self.base, 2)
        return math.log(self.base, 2) + self.p_err * math.log(self.p_err, 2) + (1-self.p_err) * math.log((1-self.p_err)/(self.base-1), 2)


    # def determine_cur_radius(self, min_candidate_error, last_block_index):
    #     if self.fixed_radius:
    #         return self.radius
    #     return min(self.prefix_radii[last_block_index] - min_candidate_error, self.max_block_error)

    # def _calculate_radii(self, factor=1, delta=0):
    #     avg_errors_per_block = factor * self._overall_radius_key_error() / self.num_blocks
    #     return [math.ceil(avg_errors_per_block * j) + 1 + delta for j in range(self.num_blocks + 1)]

    # def _determine_prefix_radii_old(self, timeout=None):
    #     if timeout is not None:
    #         signal.signal(signal.SIGALRM, timeout_handler)
    #         signal.alarm(timeout)
    #
    #     factor = 1
    #     delta = 0
    #
    #     while not self._check_overall_prefixes_error(factor, delta):
    #         delta += 1
    #
    #     if timeout is not None:
    #         signal.alarm(0)
    #
    #     return self._calculate_radii(factor, delta)
    #
    # def _check_overall_prefixes_error(self, factor=1, delta=0):
    #     max_prefix_errors = self._calculate_radii(factor, delta)
    #     max_block_errors = self.max_block_error
    #
    #     p = [binom.pmf(i, self.block_length, self.p_err) for i in range(min(max_prefix_errors[1], max_block_errors))]
    #     for j in range(2, self.num_blocks+1):
    #         p = [sum([p[k] * binom.pmf(i-k, self.block_length, self.p_err) for k in range(max(i - max_block_errors, 0), min(len(p), i+1))]) for i in range(max_prefix_errors[j])]
    #         p_sum = sum(p)
    #         if p_sum < self.success_rate:
    #             return False
    #     return True

# def timeout_handler(signum, frame):
#     print("Radii calculation out of time.")
#     raise TimeoutError("Radii calculation out of time.")