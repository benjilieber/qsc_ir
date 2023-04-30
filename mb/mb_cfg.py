import math
from enum import Enum

import numpy as np
from scipy.stats import binom

import util
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

    def __init__(self,
                 orig_cfg=None,
                 q=None,
                 block_length=None,
                 num_blocks=None,
                 p_err=0,
                 success_rate=1.0,
                 prefix_radii=None,
                 radius_picking=None,
                 full_rank_encoding=True,
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
        super().__init__(orig_cfg=orig_cfg,
                         q=q,
                         N=num_blocks * block_length,
                         p_err=p_err,
                         code_strategy=CodeStrategy.mb,
                         raw_results_file_path=raw_results_file_path,
                         agg_results_file_path=agg_results_file_path,
                         verbosity=verbosity)

        self.success_rate = success_rate
        self.block_length = block_length  # block length
        self.num_blocks = num_blocks  # number of blocks in private key
        self.N = num_blocks * block_length
        self.full_rank_encoding = full_rank_encoding
        self.use_zeroes_in_encoding_matrix = use_zeroes_in_encoding_matrix if self.q > 2 else True  # Whether to enable zeroes in encoding matrix
        self.list_size = goal_candidates_num or block_length ** 2
        self.check_length = int(math.floor(math.log(self.list_size, self.q)))
        self.indices_to_encode_strategy = indices_to_encode_strategy
        self.rounding_strategy = rounding_strategy
        self.pruning_strategy = pruning_strategy
        self.max_num_indices_to_encode = max_num_indices_to_encode or num_blocks
        self.max_candidates_num = max_candidates_num
        self.encoding_sample_size = encoding_sample_size
        self.radius_picking = radius_picking
        if self.p_err == 0.0:
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
        specific_dict = {"mb_desired_success_rate": self.success_rate,
                         "mb_block_length": self.block_length,
                         "mb_num_blocks": self.num_blocks,
                         "mb_full_rank_encoding": self.full_rank_encoding,
                         "mb_use_zeroes_in_encoding_matrix": self.use_zeroes_in_encoding_matrix,
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
                ceil_m = math.ceil(
                    math.log(total_success_prob, ceil_cdf / floor_cdf) - self.num_blocks * math.log(floor_cdf,
                                                                                                    ceil_cdf / floor_cdf))
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
            radii[i] = int(binom.ppf(success_rate_per_block, (i + 1) * self.block_length, self.p_err))

        return radii

    def determine_cur_radius(self, last_block_index):
        if self.fixed_radius:
            return self.radius
        return self.max_block_error[last_block_index]

    def _determine_num_encodings_list(self):
        total_checks = util.required_checks(self.N, self.q, self.p_err)
        num_encodings_list = []
        tot_num_encodings = 0
        for i in range(self.num_blocks):
            cur_num_encodings = math.floor(total_checks * (i + 1) / self.num_blocks - tot_num_encodings)
            num_encodings_list.append(cur_num_encodings)
            tot_num_encodings += cur_num_encodings
        return num_encodings_list


def specific_log_header():
    return ["mb_desired_success_rate",
            "mb_block_length",
            "mb_num_blocks",
            "mb_full_rank_encoding",
            "mb_use_zeroes_in_encoding_matrix",
            "mb_indices_to_encode_strategy",
            "mb_rounding_strategy",
            "mb_pruning_strategy",
            "mb_max_num_indices_to_encode",
            "mb_max_candidates_num",
            "mb_encoding_sample_size",
            "mb_radius_picking",
            "mb_predetermined_number_of_encodings"]


def specific_log_header_params():
    return ["mb_desired_success_rate",
            "mb_block_length",
            "mb_num_blocks",
            "mb_full_rank_encoding",
            "mb_use_zeroes_in_encoding_matrix",
            "mb_indices_to_encode_strategy",
            "mb_rounding_strategy",
            "mb_pruning_strategy",
            "mb_max_num_indices_to_encode",
            "mb_max_candidates_num",
            "mb_encoding_sample_size",
            "mb_radius_picking",
            "mb_predetermined_number_of_encodings"]
