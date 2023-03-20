import math
import os
import time
from timeit import default_timer as timer

import numpy as np
import scipy
from scipy.stats import entropy

from mb.alice import Alice
from mb.bob import Bob
from mb.encoder import Encoder
from mb.linear_code_generator import LinearCodeGenerator
from mb.mb_cfg import IndicesToEncodeStrategy, LinearCodeFormat
from protocol import Protocol


class MbProtocol(Protocol):
    def __init__(self, cfg, a, b):
        super().__init__(cfg=cfg, a=a, b=b)

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
                self.run_single_round(encode_new_block=False, goal_list_size=self.cfg.list_size)
        end = timer()
        a_guess_list = [[int(a_guess_val) for a_guess_block in a_guess for a_guess_val in a_guess_block] for a_guess in
                        self.bob.a_candidates]
        return self.get_results(key_length=self.cfg.N,  # in q-ary bits
                                a_guess_list=a_guess_list,
                                a_key=self.a,
                                b_key_list=a_guess_list,
                                leak_size=self.total_leak * math.log2(self.cfg.q),  # in bits
                                matrix_size=self.matrix_communication_size,  # in bits
                                bob_communication_size=self.bob_communication_size,  # in bits
                                time=end - start)

    def run_single_round(self, encode_new_block, goal_list_size=None):
        encoded_blocks_indices = self.pick_indices_to_encode(self.num_candidates_per_block, encode_new_block)
        number_of_encodings = self.determine_number_of_encodings(self.cur_candidates_num, encode_new_block,
                                                                 encoded_blocks_indices[-1],
                                                                 goal_list_size=goal_list_size)
        encoding_format = self.determine_encoding_format(encode_new_block, encoded_blocks_indices, number_of_encodings)

        encoding_matrix_prefix = self.pick_encoding_matrix_prefix(encode_new_block, encoded_blocks_indices,
                                                                  number_of_encodings, encoding_format)
        encoding = self.alice.encode_blocks(encode_new_block, encoded_blocks_indices, number_of_encodings,
                                            encoding_matrix_prefix, encoding_format)
        [self.cur_candidates_num, self.num_candidates_per_block] = self.bob.decode_multi_block(encoded_blocks_indices,
                                                                                               encoding)

        self.update_communication_stats(encoding)
        self.candidates_num_history.append(self.cur_candidates_num)
        self.num_encodings_history.append(number_of_encodings)
        self.num_encoded_blocks_history.append(len(encoded_blocks_indices))

    def pick_indices_to_encode(self, num_candidates_per_block, encode_new_block):
        if encode_new_block:
            num_candidates_per_block = num_candidates_per_block + [math.inf]
        if self.cfg.indices_to_encode_strategy == IndicesToEncodeStrategy.all_multi_candidate_blocks or not encode_new_block:
            return [i for i, e in enumerate(num_candidates_per_block) if e > 1]
        if self.cfg.indices_to_encode_strategy == IndicesToEncodeStrategy.most_candidate_blocks:
            num_indices_to_encode = min(self.cfg.max_num_indices_to_encode,
                                        len(num_candidates_per_block) - num_candidates_per_block.count(1))
            return sorted(range(len(num_candidates_per_block)), key=lambda sub: num_candidates_per_block[sub])[
                   -num_indices_to_encode:]

    def determine_number_of_encodings(self, cur_candidates_num, encode_new_block, last_block_index,
                                      goal_list_size=None):
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
                    [scipy.special.comb(self.cfg.block_length, i) * (self.cfg.q - 1) ** (self.cfg.block_length - i) for
                     i in range(r + 1)]), self.cfg.q)

            # old version of encoding-number picking
            required_number_of_encodings_raw = max(self.cfg.round(
                complement_space_size_log - math.log(self.cfg.list_size,
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
            return min(max(self.cfg.round(math.log(cur_candidates_num / goal_list_size, self.cfg.q)), 1),
                       self.cfg.block_length)
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
                expected_matrix_complexity_log = math.log(sum([scipy.special.binom(self.cfg.block_length, err) * (
                        self.cfg.q - 1) ** (self.cfg.block_length - err) for err in range(
                    self.cfg.determine_cur_radius(latest_block_index) + 1)]))
            # expected_num_buckets = np.unique([candidate[indices_to_encode] for candidate in self.bob.a_candidates], axis=0)
            expected_prefix_num_buckets_log = min(math.log(self.cur_candidates_num), np.sum(
                [math.log(self.num_candidates_per_block[i]) for i in indices_to_encode[:-1]]),
                                                  number_of_encodings * math.log(self.cfg.q))
            expected_affine_subspace_complexity_log = expected_prefix_num_buckets_log + (
                    self.cfg.block_length - number_of_encodings) * math.log(self.cfg.q)

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
            cur_encoding_matrix_prefix_option = np.zeros([prefix_length, self.cfg.block_length, number_of_encodings],
                                                         dtype=int)
            for i, block_index in enumerate(blocks_indices[:prefix_length]):
                cur_encoding_matrix_delta_option = self.linear_code_generator.generate_encoding_matrix(
                    number_of_encodings)
                cur_encoding_matrix_prefix_option[i] = cur_encoding_matrix_delta_option
            encoding_matrix_prefix_options.append(cur_encoding_matrix_prefix_option)
        if self.cfg.encoding_sample_size == 1:
            return encoding_matrix_prefix_options[0]
        else:
            cur_encoding_matrix_prefix_unique_counts = [np.unique(
                [self.encoder.encode_multi_block(cur_encoding_matrix_prefix,
                                                 np.take(a_candidate, blocks_indices[:prefix_length], axis=0)) for
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
            self.matrix_communication_size += (
                                                      encoding.prefix_encoding_matrix.size + encoding.image_base_source.size + encoding.kernel_base.size) * math.log(
                self.cfg.q, 2)
            self.total_leak += len(encoding.image_base_source)
        else:
            pass
        self.bob_communication_size += math.log(self.cur_candidates_num or 1, 2) + \
                                       sum([math.log(i or 1, 2) + math.log(num or 1, 2) for i, num in
                                            enumerate(self.num_candidates_per_block)])
        self.total_communication_size = self.matrix_communication_size + self.total_leak * math.log(self.cfg.q, 2) + \
                                        self.bob_communication_size
