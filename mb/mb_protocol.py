import math
import os
import time

import numpy as np
import scipy

import util
from mb.alice import Alice
from mb.bob import Bob
from mb.encoder import Encoder
from mb.mb_cfg import IndicesToEncodeStrategy, LinearCodeFormat
from result import Result, Status
from timeit import default_timer as timer
from scipy.stats import entropy
from mb.linear_code_generator import LinearCodeGenerator

class MbProtocol(object):
    def __init__(self, cfg, a, b):
        self.cfg = cfg
        self.a = a
        self.b = b
        self.alice = Alice(cfg, a)
        self.bob = Bob(cfg, b)

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.linear_code_generator = LinearCodeGenerator(cfg)
        self.encoder = Encoder(cfg.q, cfg.block_length)

        self.total_leak = 0
        self.matrix_communication_size = 0
        self.bob_communication_size = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1  # must be initialized to 1 for encoding-picking function to work as expected, if I change this ensure it doesn't break
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

            if self.cfg.max_candidates_num and (self.cur_candidates_num > self.cfg.max_candidates_num):
                self.run_single_round(encode_new_block=False, goal_list_size=self.cfg.goal_candidates_num)
        end = timer()

        ml_result = self.get_result(time=end-start, is_ml=True)

        if self.cfg.verbosity and self.cur_candidates_num > 0:
            print("hamming distance x and most probable x', pre-reducing: " + str(
                util.hamming_multi_block(self.bob.a_candidates[np.argmin(self.bob.a_candidates_errors)], self.alice.a)))

        while self.cur_candidates_num > 1:
            self.run_single_round(encode_new_block=False)

        if self.cfg.verbosity:
            print("error_blocks_indices: " + str([sum(a_block == b_block) for a_block, b_block in zip(self.alice.a, self.bob.b)]))
            print("candidates_num_history: " + str(self.candidates_num_history))
            print("candidates_buckets_num_history: " + str(self.bob.candidates_buckets_num_history))
            print("mean_avg_candidates_per_img: " + str(sum(self.bob.avg_candidates_per_img_history)/len(self.bob.avg_candidates_per_img_history)) + ", avg_candidates_per_img_history: " + str(self.bob.avg_candidates_per_img_history))
            print("mean_pruning_rate: " + str(1-np.product([1 - pr for pr in self.bob.pruning_rate_history])**(1/(len(self.bob.pruning_rate_history) or 1))) + ", pruning_rate_history: " + str(self.bob.pruning_rate_history))
            print("pruning_fail_prob_history: " + str(self.bob.pruning_fail_prob_history))
            print("radii_list: " + str(self.r_list))
            print("num_encodings_history: " + str(self.num_encodings_history) + ", total: " + str(sum(self.num_encodings_history)) + " (theoretic: " + str(
                util.required_checks(self.cfg.N, self.cfg.q, self.cfg.p_err)) + ")")
            print("num_encoded_blocks_history: " + str(self.num_encoded_blocks_history))
        non_ml_result = self.get_result(time=end-start, is_ml=False)

        return [non_ml_result, ml_result]

    def run_single_round(self, encode_new_block, goal_list_size=None):
        encoded_blocks_indices = self.pick_indices_to_encode(self.num_candidates_per_block, encode_new_block)
        number_of_encodings = self.determine_number_of_encodings(self.cur_candidates_num, encode_new_block,
                                                                 encoded_blocks_indices[-1], goal_list_size=goal_list_size)
        encoding_format = self.determine_encoding_format(encode_new_block, encoded_blocks_indices, number_of_encodings)
        # if self.cfg.verbosity:
        #     print(encoding_format)

        encoding_matrix_prefix = self.pick_encoding_matrix_prefix(encode_new_block, encoded_blocks_indices, number_of_encodings, encoding_format)
        encoding = self.alice.encode_blocks(encode_new_block, encoded_blocks_indices, number_of_encodings, encoding_matrix_prefix, encoding_format)
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

    def determine_number_of_encodings(self, cur_candidates_num, encode_new_block, last_block_index, goal_list_size=None):
        if encode_new_block:
            r = self.cfg.determine_cur_radius(last_block_index)
            self.r_list.append(r)
            if self.cfg.fixed_number_of_encodings:
                return self.cfg.number_of_encodings_list[last_block_index]
            if r == 0:
                complement_space_size_log = self.cfg.block_length * math.log(self.cfg.q - 1, self.cfg.q)
            elif r == self.cfg.block_length:
                complement_space_size_log = self.cfg.block_length * math.log(self.cfg.q, self.cfg.q)
            else:
                complement_space_size_log = math.log(sum(
                    [scipy.special.comb(self.cfg.block_length, i) * (self.cfg.q - 1) ** (self.cfg.block_length - i) for i in range(r + 1)]), self.cfg.q)

            # old version of encoding-number picking
            required_number_of_encodings_raw = max(self.cfg.round(
                complement_space_size_log - math.log(self.cfg.goal_candidates_num,
                                                     self.cfg.q) + math.log(
                    self.cur_candidates_num, self.cfg.q)), 1)
            required_number_of_encodings = min(required_number_of_encodings_raw, self.cfg.block_length)
            return required_number_of_encodings

            # new version of encoding-number picking
            # if cur_candidates_num == 1:
            #     required_number_of_encodings_raw = math.log(complement_space_size, self.cfg.base) - math.log(self.cfg.goal_candidates_num-1, self.cfg.base)
            # else:
            #     expected_candidate_space_size = complement_space_size * cur_candidates_num
            #     required_number_of_encodings_raw = math.log(expected_candidate_space_size, self.cfg.base) + math.log(1+math.sqrt(1+4*self.cfg.goal_candidates_num/expected_candidate_space_size), self.cfg.base) - math.log(2, self.cfg.base) - math.log(
            #         self.cfg.goal_candidates_num - 1, self.cfg.base)
            # required_number_of_encodings = min(max(self.cfg.round(required_number_of_encodings_raw), 1), self.cfg.block_length)
            #
            # return required_number_of_encodings

        else:
            goal_list_size = goal_list_size or 1
            return min(max(self.cfg.round(math.log(cur_candidates_num / goal_list_size, self.cfg.q)), 1), self.cfg.block_length)
            # old version of encoding-number picking
            # return min(max(math.floor(math.log(cur_candidates_num, self.cfg.base)), 1), self.cfg.block_length_hash_base)

    def determine_encoding_format(self, encode_new_block, indices_to_encode, number_of_encodings):
        if not encode_new_block:
            return LinearCodeFormat.MATRIX

        else:
            latest_block_index = indices_to_encode[-1]
            if self.cfg.determine_cur_radius(latest_block_index) == self.cfg.block_length:
                return LinearCodeFormat.AFFINE_SUBSPACE
            if self.cfg.determine_cur_radius(latest_block_index) == 0:
                expected_matrix_complexity_log = self.cfg.block_length * math.log(self.cfg.q - 1)
            else:
                expected_matrix_complexity_log = math.log(sum([scipy.special.binom(self.cfg.block_length, err) * (self.cfg.q - 1) ** (self.cfg.block_length - err) for err in range(self.cfg.determine_cur_radius(latest_block_index) + 1)]))
            # expected_num_buckets = np.unique([candidate[indices_to_encode] for candidate in self.bob.a_candidates], axis=0)
            expected_prefix_num_buckets_log = min(math.log(self.cur_candidates_num), np.sum([math.log(self.num_candidates_per_block[i]) for i in indices_to_encode[:-1]]), number_of_encodings * math.log(self.cfg.q))
            expected_affine_subspace_complexity_log = expected_prefix_num_buckets_log + (self.cfg.block_length - number_of_encodings) * math.log(self.cfg.q)

            if expected_matrix_complexity_log < expected_affine_subspace_complexity_log:
                return LinearCodeFormat.MATRIX
            else:
                return LinearCodeFormat.AFFINE_SUBSPACE

    def pick_encoding_matrix_prefix(self, encode_new_block, blocks_indices, number_of_encodings, encoding_format):
        prefix_length = len(blocks_indices) - (encode_new_block or encoding_format == LinearCodeFormat.AFFINE_SUBSPACE)

        # encoding_matrix_prefix = np.zeros(
        #     [prefix_length, self.cfg.block_length_hash_base, number_of_encodings], dtype=int)
        # for i, block_index in enumerate(blocks_indices[:prefix_length]):
        #     if self.cfg.encoding_sample_size == 1:
        #         cur_encoding_matrix_delta = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
        #         encoding_matrix_prefix[i] = cur_encoding_matrix_delta
        #     else:
        #         cur_encoding_matrix_delta_options = [
        #             self.linear_code_generator.generate_encoding_matrix(number_of_encodings) for _ in
        #             range(self.cfg.encoding_sample_size)]
        #         cur_encoding_matrix_delta_unique_counts = [np.unique(
        #             [self.encoder.encode_single_block(cur_encoding_matrix_delta, a_block_candidate) for
        #              a_block_candidate in self.bob.a_candidates_per_block[block_index]], axis=0, return_counts=True)[1]
        #                                                    for cur_encoding_matrix_delta in
        #                                                    cur_encoding_matrix_delta_options]
        #         cur_encoding_matrix_delta_entropies = [entropy(unique_counts) for unique_counts in
        #                                                cur_encoding_matrix_delta_unique_counts]
        #         encoding_matrix_prefix[i] = cur_encoding_matrix_delta_options[
        #             np.argmax(cur_encoding_matrix_delta_entropies)]
        # unique_count = np.unique(
        #         [self.encoder.encode_multi_block(encoding_matrix_prefix, np.take(a_candidate, blocks_indices[:prefix_length], axis=0)) for
        #          a_candidate in self.bob.a_candidates], axis=0, return_counts=True)[1]
        # print(unique_count)
        # print(entropy(unique_count))

        encoding_matrix_prefix_options = []
        for _ in range(self.cfg.encoding_sample_size):
            cur_encoding_matrix_prefix_option = np.zeros([prefix_length, self.cfg.block_length, number_of_encodings], dtype=int)
            for i, block_index in enumerate(blocks_indices[:prefix_length]):
                cur_encoding_matrix_delta_option = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
                cur_encoding_matrix_prefix_option[i] = cur_encoding_matrix_delta_option
            encoding_matrix_prefix_options.append(cur_encoding_matrix_prefix_option)
        if self.cfg.encoding_sample_size == 1:
            return encoding_matrix_prefix_options[0]
        else:
            cur_encoding_matrix_prefix_unique_counts = [np.unique(
                [self.encoder.encode_multi_block(cur_encoding_matrix_prefix, np.take(a_candidate, blocks_indices[:prefix_length], axis=0)) for
                 a_candidate in self.bob.a_candidates], axis=0, return_counts=True)[1]
                                                       for cur_encoding_matrix_prefix in
                                                       encoding_matrix_prefix_options]
            # print(cur_encoding_matrix_prefix_unique_counts)
            cur_encoding_matrix_prefix_entropies = [entropy(unique_counts) for unique_counts in
                                                   cur_encoding_matrix_prefix_unique_counts]
            # print(cur_encoding_matrix_prefix_entropies)
            return encoding_matrix_prefix_options[np.argmax(cur_encoding_matrix_prefix_entropies)]

        # encoding_matrix_prefix = np.zeros(
        #     [prefix_length, self.cfg.block_length_hash_base, number_of_encodings], dtype=int)
        # for i, block_index in enumerate(blocks_indices[:prefix_length]):
        #     if self.cfg.encoding_sample_size == 1:
        #         cur_encoding_matrix_delta = self.linear_code_generator.generate_encoding_matrix(number_of_encodings)
        #         encoding_matrix_prefix[i] = cur_encoding_matrix_delta
        #     else:
        #         cur_encoding_matrix_delta_options = [
        #             self.linear_code_generator.generate_encoding_matrix(number_of_encodings) for _ in
        #             range(self.cfg.encoding_sample_size)]
        #         cur_encoding_matrix_delta_unique_counts = [np.unique(
        #             [self.encoder.encode_single_block(cur_encoding_matrix_delta, a_block_candidate) for
        #              a_block_candidate in self.bob.a_candidates_per_block[block_index]], axis=0, return_counts=True)[1]
        #                                                    for cur_encoding_matrix_delta in
        #                                                    cur_encoding_matrix_delta_options]
        #         print(cur_encoding_matrix_delta_unique_counts)
        #         cur_encoding_matrix_delta_entropies = [entropy(unique_counts) for unique_counts in
        #                                                cur_encoding_matrix_delta_unique_counts]
        #         print(cur_encoding_matrix_delta_entropies)
        #         encoding_matrix_prefix[i] = cur_encoding_matrix_delta_options[
        #             np.argmax(cur_encoding_matrix_delta_entropies)]
        return encoding_matrix_prefix

    def update_communication_stats(self, encoding):
        if encoding.format == LinearCodeFormat.MATRIX:
            self.matrix_communication_size += encoding.encoding_matrix.size * math.log(self.cfg.q, 2)
            self.total_leak += len(encoding.encoded_vector)
        elif encoding.format == LinearCodeFormat.AFFINE_SUBSPACE:
            self.matrix_communication_size += (encoding.prefix_encoding_matrix.size + encoding.image_base_source.size + encoding.kernel_base.size) * math.log(self.cfg.q, 2)
            self.total_leak += len(encoding.image_base_source)
        else:
            pass
        self.bob_communication_size += math.log(self.cur_candidates_num or 1, 2) + \
                                       sum([math.log(i or 1, 2) + math.log(num or 1, 2) for i, num in
                                            enumerate(self.num_candidates_per_block)])
        self.total_communication_size = self.matrix_communication_size + self.total_leak * math.log(self.cfg.q, 2) + \
                                        self.bob_communication_size

    def get_result(self, time, is_ml):
        if self.cur_candidates_num:
            assert ((self.cur_candidates_num == 1) or is_ml)
            a_guess = self.bob.a_candidates[np.argmin(self.bob.a_candidates_errors)]
            ser = self.get_ser(a_guess)
            if ser == 0.0:
                return Result(cfg=self.cfg, with_ml=is_ml, result_status=Status.success, ser_completed_only=0.0,
                              key_rate=self.calculate_key_rate(final=False),
                              key_rate_success_only=self.calculate_key_rate(final=False),
                              key_rate_completed_only=self.calculate_key_rate(final=False),
                              leak_rate=self.calculate_key_rate(final=False),
                              leak_rate_completed_only=self.total_leak / self.cfg.N,
                              leak_rate_success_only=self.total_leak / self.cfg.N,
                              matrix_size_rate=self.matrix_communication_size / self.cfg.N,
                              matrix_size_rate_success_only=self.matrix_communication_size / self.cfg.N,
                              bob_communication_rate=self.bob_communication_size / self.cfg.N,
                              bob_communication_rate_success_only=self.bob_communication_size / self.cfg.N,
                              total_communication_rate=self.total_communication_size / self.cfg.N,
                              total_communication_rate_success_only=self.total_communication_size / self.cfg.N,
                              time_rate=time / self.cfg.N)
            else:
                return Result(cfg=self.cfg, with_ml=is_ml, result_status=Status.fail, ser_completed_only=ser,
                              ser_fail_only=ser,
                              key_rate=self.calculate_key_rate(final=False),
                              key_rate_completed_only=self.calculate_key_rate(final=False),
                              leak_rate=self.calculate_key_rate(final=False),
                              leak_rate_completed_only=self.total_leak / self.cfg.N,
                              matrix_size_rate=self.matrix_communication_size / self.cfg.N,
                              bob_communication_rate=self.bob_communication_size / self.cfg.N,
                              total_communication_rate=self.total_communication_size / self.cfg.N,
                              time_rate=time / self.cfg.N)
        else:
            return Result(cfg=self.cfg, with_ml=is_ml, result_status=Status.abort,
                          key_rate=self.calculate_key_rate(final=False),
                          leak_rate=self.calculate_key_rate(final=False),
                          matrix_size_rate=self.matrix_communication_size / self.cfg.N,
                          bob_communication_rate=self.bob_communication_size / self.cfg.N,
                          total_communication_rate=self.total_communication_size / self.cfg.N,
                          time_rate=time / self.cfg.N)

    def calculate_key_rate(self, final=True):
        if final and (not self.is_success()):
            return 0
        key_size = (self.cfg.N - self.total_leak) * math.log(self.cfg.q, 2)
        return key_size / self.cfg.N

    def get_ser(self, a_guess):
        return util.hamming_multi_block(a_guess, self.alice.a) / self.cfg.N

    def get_status(self, a_guess=None):
        if (a_guess is None) and (self.cur_candidates_num == 0):
            return Status.abort
        assert ((a_guess is not None) or (len(self.bob.a_candidates) == 1))
        a_guess = a_guess if (a_guess is not None) else self.bob.a_candidates[0]
        if self.get_ser(a_guess) == 0.0:
            return Status.success
        else:
            return Status.fail

    def is_success(self, a_guess=None):
        return self.get_status(a_guess=a_guess) == Status.success