import math
import os
import time

import numpy as np
import scipy

import util
from alice import Alice
from bob import Bob
from protocol_configs import IndicesToEncodeStrategy, LinearCodeFormat
from result import Result
from timeit import default_timer as timer


class MultiBlockProtocol(object):
    def __init__(self, protocol_cfg, a, b):
        self.cfg = protocol_cfg
        self.a = a
        self.b = b
        self.alice = Alice(protocol_cfg, a)
        self.bob = Bob(protocol_cfg, b)

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.total_encoding_size = 0
        self.matrix_communication_size = 0
        self.bob_communication_size = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1
        self.num_candidates_per_block = []

        self.candidates_num_history = []
        self.r_list = []
        self.num_encodings_history = []
        self.num_encoded_blocks_history = []

    def run(self):
        start = timer()
        for block_index in range(self.cfg.num_blocks):
            self.run_single_round(encode_new_block=True)

            if self.cur_candidates_num == 0:
                break

            if self.cfg.upper_threshold and (self.cur_candidates_num > self.cfg.upper_threshold):
                self.run_single_round(encode_new_block=False)

        while self.cur_candidates_num > 1:
            self.run_single_round(encode_new_block=False)

        end = timer()

        print("error_blocks_indices: " + str([sum(a_block == b_block) for a_block, b_block in zip(self.alice.a, self.bob.b)]))
        print("candidates_num_history: " + str(self.candidates_num_history))
        print("candidates_buckets_num_history: " + str(self.bob.candidates_buckets_num_history))
        print("mean_avg_candidates_per_img: " + str(sum(self.bob.avg_candidates_per_img_history)/len(self.bob.avg_candidates_per_img_history)) + ", avg_candidates_per_img_history: " + str(self.bob.avg_candidates_per_img_history))
        print("mean_pruning_rate: " + str(1-np.product([1 - pr for pr in self.bob.pruning_rate_history])**(1/(len(self.bob.pruning_rate_history) or 1))) + ", pruning_rate_history: " + str(self.bob.pruning_rate_history))
        print("radii_list: " + str(self.r_list))
        print("num_encodings_history: " + str(self.num_encodings_history) + ", total: " + str(sum(self.num_encodings_history)) + " (theoretic: " + str(util.required_checks(self.cfg.key_length, self.cfg.base, self.cfg.p_err)) + ")")
        print("num_encoded_blocks_history: " + str(self.num_encoded_blocks_history))

        return Result(cfg=self.cfg, is_success=self.is_success(), key_rate=self.calculate_key_rate(),
                      encoding_size_rate=self.total_encoding_size / self.cfg.key_length,
                      matrix_size_rate=self.matrix_communication_size / self.cfg.key_length,
                      bob_communication_rate=self.bob_communication_size / self.cfg.key_length,
                      total_communication_rate=self.total_communication_size / self.cfg.key_length,
                      time_rate=(end - start) / self.cfg.key_length)

    def run_single_round(self, encode_new_block):
        encoded_blocks_indices = self.pick_indices_to_encode(self.num_candidates_per_block, encode_new_block)
        number_of_encodings = self.determine_number_of_encodings(self.cur_candidates_num, encode_new_block,
                                                                 encoded_blocks_indices[-1])
        encoding_format = self.determine_encoding_format(encode_new_block, encoded_blocks_indices, number_of_encodings)
        # print(encoding_format)

        encoding = self.alice.encode_blocks(encoded_blocks_indices, number_of_encodings, encoding_format)
        [self.cur_candidates_num, self.num_candidates_per_block] = self.bob.decode_multi_block(encoded_blocks_indices, encoding)

        self.update_communication_stats(encoding)
        self.candidates_num_history.append(self.cur_candidates_num)
        self.num_encodings_history.append(number_of_encodings)
        self.num_encoded_blocks_history.append(len(encoded_blocks_indices))

        # print(min([util.hamming_multi_block(a_candidate, self.alice.a, False) for a_candidate in self.bob.a_candidates]))
        # if all([util.hamming_multi_block(a_candidate, self.alice.a, False) for a_candidate in self.bob.a_candidates]):
        #     print("lost the right candidate at block: " + str(encoded_blocks_indices[-1]))
        #     print("number of candidates: " + str(len(self.bob.a_candidates)))
        #     # print("max number of errors at this stage: " + str(max([util.closeness_multi_block(a_candidate, self.bob.b, False) for a_candidate in self.bob.a_candidates])))
        #     # print("actual number of errors at this stage: " + str(util.closeness_multi_block(self.alice.a[:encoded_blocks_indices[-1]], self.bob.b, False)))


    def pick_indices_to_encode(self, num_candidates_per_block, encode_new_block):
        if encode_new_block:
            num_candidates_per_block = num_candidates_per_block + [math.inf]
        if self.cfg.indices_to_encode_strategy == IndicesToEncodeStrategy.ALL_MULTI_CANDIDATE_BLOCKS or not encode_new_block:
            return [i for i, e in enumerate(num_candidates_per_block) if e > 1]
        if self.cfg.indices_to_encode_strategy == IndicesToEncodeStrategy.MOST_CANDIDATE_BLOCKS:
            num_indices_to_encode = min(self.cfg.max_num_indices_to_encode,
                                        len(num_candidates_per_block) - num_candidates_per_block.count(1))
            return sorted(range(len(num_candidates_per_block)), key=lambda sub: num_candidates_per_block[sub])[
                   -num_indices_to_encode:]

    def determine_number_of_encodings(self, cur_candidates_num, encode_new_block, last_block_index):
        if encode_new_block:
            r = self.cfg.determine_cur_radius(last_block_index)
            self.r_list.append(r)
            if self.cfg.fixed_number_of_encodings:
                return self.cfg.number_of_encodings_list[last_block_index]
            complement_space_size = sum(
                [scipy.special.comb(self.cfg.block_length, i) * (self.cfg.base-1) ** (self.cfg.block_length - i) for i in range(r + 1)])
            required_number_of_encodings_raw = max(math.ceil(
                math.log(complement_space_size, self.cfg.base) - math.log(self.cfg.max_candidates_num,
                                                                          self.cfg.base) + math.log(
                    self.cur_candidates_num, self.cfg.base)), 1)
            return min(required_number_of_encodings_raw, self.cfg.block_length_hash_base)

        else:
            return min(max(math.floor(math.log(cur_candidates_num, self.cfg.base)), 1), self.cfg.block_length_hash_base)

    def determine_encoding_format(self, encode_new_block, indices_to_encode, number_of_encodings):
        if not encode_new_block:
            return LinearCodeFormat.MATRIX

        else:
            latest_block_index = indices_to_encode[-1]
            expected_matrix_complexity = sum([scipy.special.binom(self.cfg.block_length, err) * (self.cfg.base - 1)**(self.cfg.block_length - err) for err in range(self.cfg.determine_cur_radius(latest_block_index) + 1)])
            # expected_num_buckets = np.unique([candidate[indices_to_encode] for candidate in self.bob.a_candidates], axis=0)
            expected_prefix_num_buckets = min(self.cur_candidates_num, np.product([self.num_candidates_per_block[i] for i in indices_to_encode[:-1]]), self.cfg.base ** number_of_encodings)
            expected_affine_subspace_complexity = expected_prefix_num_buckets * (self.cfg.base ** (self.cfg.block_length - number_of_encodings))

            if expected_matrix_complexity < expected_affine_subspace_complexity:
                return LinearCodeFormat.MATRIX
            else:
                return LinearCodeFormat.AFFINE_SUBSPACE


    def update_communication_stats(self, encoding):
        if encoding.format == LinearCodeFormat.MATRIX:
            self.matrix_communication_size += encoding.encoding_matrix.size * math.log(self.cfg.base, 2)
            self.total_encoding_size += len(encoding.encoded_vector)
        elif encoding.format == LinearCodeFormat.AFFINE_SUBSPACE:
            self.matrix_communication_size += (encoding.prefix_encoding_matrix.size + encoding.image_base_source.size + encoding.kernel_base.size) * math.log(self.cfg.base, 2)
            self.total_encoding_size += len(encoding.image_base_source)
        else:
            pass
        self.bob_communication_size += math.log(self.cur_candidates_num or 1, 2) + \
                                       sum([math.log(i or 1, 2) + math.log(num or 1, 2) for i, num in
                                            enumerate(self.num_candidates_per_block)])
        self.total_communication_size = self.matrix_communication_size + self.total_encoding_size * math.log(self.cfg.base, 2) + \
                                        self.bob_communication_size

    def calculate_key_rate(self):
        if self.cur_candidates_num == 0 or not self.is_success():
            return 0
        key_size = (self.cfg.key_length - self.total_encoding_size) * math.log(self.cfg.base, 2)
        return key_size / self.cfg.key_length

    def is_success(self):
        return (len(self.bob.a_candidates) == 1) and (
                util.hamming_multi_block(self.bob.a_candidates[0], self.alice.a) == 0)
