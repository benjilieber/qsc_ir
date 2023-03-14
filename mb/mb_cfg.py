import math
from enum import Enum
from scipy.stats import binom
import util
import numpy as np
from cfg import Cfg, CodeStrategy

class IndicesToEncodeStrategy(Enum):
    all_multi_candidate_blocks = "all_multi_candidate_blocks"
    most_candidate_blocks = "most_candidate_blocks"

    def __str__(self):
        return self.value

class RoundingStrategy(Enum):
    floor = 'floor'
    ceil = 'ceil'

    def __str__(self):
        return self.value

class PruningStrategy(Enum):
    radii_probabilities = 'radii_probabilities'
    relative_weights = 'relative_weights'

    def __str__(self):
        return self.value

class LinearCodeFormat(Enum):
    MATRIX = 'matrix'
    AFFINE_SUBSPACE = 'affine_subspace'

    def __str__(self):
        return self.value

class MbCfg(Cfg):

    def __init__(self, orig_cfg=None, q=None, block_length=None, num_blocks=None, p_err=0, success_rate=1.0, prefix_radii=None, radius_picking=None, full_rank_encoding=True,
                 use_zeroes_in_encoding_matrix=True,
                 goal_candidates_num=None,
                 max_candidates_num=None,
                 indices_to_encode_strategy=IndicesToEncodeStrategy.all_multi_candidate_blocks,
                 rounding_strategy=RoundingStrategy.ceil,
                 pruning_strategy=PruningStrategy.radii_probabilities,
                 fixed_number_of_encodings=None,
                 max_num_indices_to_encode=None,
                 encoding_sample_size=1,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        self.code_strategy = CodeStrategy.mb
        super().__init__(orig_cfg=orig_cfg, q=q, N=num_blocks * block_length, p_err=p_err, raw_results_file_path=raw_results_file_path, agg_results_file_path=agg_results_file_path, verbosity=verbosity)

        self.success_rate = success_rate
        self.block_length = block_length  # block length
        self.num_blocks = num_blocks  # number of blocks in private key
        self.N = num_blocks * block_length
        self.full_rank_encoding = full_rank_encoding
        self.use_zeroes_in_encoding_matrix = use_zeroes_in_encoding_matrix if self.q > 2 else True  # Whether to enable zeroes in encoding matrix
        self.goal_candidates_num = goal_candidates_num or block_length ** 2
        self.indices_to_encode_strategy = indices_to_encode_strategy
        self.rounding_strategy = rounding_strategy
        self.pruning_strategy = pruning_strategy
        self.max_num_indices_to_encode = max_num_indices_to_encode or num_blocks
        self.max_candidates_num = max_candidates_num
        self.encoding_sample_size = encoding_sample_size
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
            self.pruning_success_rate = self.success_rate
        else:
            self.fixed_radius = False
            self.max_block_error = self._radius_for_max_block_error()
            self.prefix_radii = prefix_radii or self._determine_prefix_radii()
            self.pruning_success_rate = math.sqrt(self.success_rate)
            # self.prefix_radii = prefix_radii or self._determine_prefix_radii(timeout)

        self.fixed_number_of_encodings = fixed_number_of_encodings
        if fixed_number_of_encodings:
            self.number_of_encodings_list = self._determine_num_encodings_list()

    def log_dict(self):
        super_dict = super().log_dict()
        specific_dict = {"code_strategy": str(self.code_strategy),
                         "mb_success_rate": self.success_rate,
                         "mb_block_length": self.block_length,
                         "mb_num_blocks": self.num_blocks,
                         "mb_full_rank_encoding": self.full_rank_encoding,
                         "mb_use_zeroes_in_encoding_matrix": self.use_zeroes_in_encoding_matrix,
                         "mb_goal_candidates_num": self.goal_candidates_num,
                         "mb_indices_to_encode_strategy": str(self.indices_to_encode_strategy),
                         "mb_rounding_strategy": str(self.rounding_strategy),
                         "mb_pruning_strategy": str(self.pruning_strategy),
                         "mb_max_num_indices_to_encode": self.max_num_indices_to_encode,
                         "mb_max_candidates_num": self.max_candidates_num,
                         "mb_encoding_sample_size": self.encoding_sample_size,
                         "mb_radius_picking": self.radius_picking,
                         "mb_predetermined_number_of_encodings": self.fixed_number_of_encodings}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return {**super_dict, **specific_dict}

    def round(self, number):
        if self.rounding_strategy == RoundingStrategy.floor:
            return math.floor(number)
        elif self.rounding_strategy == RoundingStrategy.ceil:
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
        return int(binom.ppf(self.success_rate, self.N, self.p_err))

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
        return util.closeness_multi_block(x[:num_blocks_considered], y[:num_blocks_considered]) <= self.prefix_radii[num_blocks_considered - 1]

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
        total_checks = util.required_checks(self.N, self.q, self.p_err)
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
def specific_log_header():
    return ["code_strategy",
            "mb_success_rate",
            "mb_block_length",
            "mb_num_blocks",
            "mb_full_rank_encoding",
            "mb_use_zeroes_in_encoding_matrix",
            "mb_goal_candidates_num",
            "mb_indices_to_encode_strategy",
            "mb_rounding_strategy",
            "mb_pruning_strategy",
            "mb_max_num_indices_to_encode",
            "mb_max_candidates_num",
            "mb_encoding_sample_size",
            "mb_radius_picking",
            "mb_predetermined_number_of_encodings"]
def specific_log_header_params():
    return ["code_strategy",
            "mb_success_rate",
            "mb_block_length",
            "mb_num_blocks",
            "mb_full_rank_encoding",
            "mb_use_zeroes_in_encoding_matrix",
            "mb_goal_candidates_num",
            "mb_indices_to_encode_strategy",
            "mb_rounding_strategy",
            "mb_pruning_strategy",
            "mb_max_num_indices_to_encode",
            "mb_max_candidates_num",
            "mb_encoding_sample_size",
            "mb_radius_picking",
            "mb_predetermined_number_of_encodings"]