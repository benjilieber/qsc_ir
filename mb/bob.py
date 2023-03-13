import itertools
import math
import random

import numpy as np
import scipy.special

import util

from mb.encoder import Encoder
from itertools import product

from mb.mb_cfg import PruningStrategy, LinearCodeFormat


class Bob(object):

    def __init__(self, protocol_configs, b):
        self.cfg = protocol_configs
        self.encoder = Encoder(protocol_configs.q, protocol_configs.block_length)
        self.b = np.array_split(b, protocol_configs.num_blocks)  # Bob's private key, partitioned into blocks
        self.a_candidates = np.array([])
        self.a_candidates_errors = np.array([])
        self.a_candidates_per_block = []
        self.cur_success_prob = 1.0
        # self.a_candidates_per_err_cnt = {}
        # self.min_candidate_error = 0
        self.candidates_buckets_num_history = []
        self.avg_candidates_per_img_history = []
        self.pruning_rate_history = []
        self.pruning_fail_prob_history = []

    def decode_multi_block(self, block_indices, encoding):
        """
        Bob decodes A'.
        Using B, M and A', finds all candidates for A.
        Takes into account error within radius.
        """
        latest_block_index = block_indices[-1]
        num_blocks_before = len(self.a_candidates_per_block)
        num_blocks_after = max(latest_block_index + 1, len(self.a_candidates_per_block))

        if latest_block_index >= num_blocks_before:  # if there is a new block, find new candidates for it
            latest_block_radius = self.cfg.determine_cur_radius(latest_block_index)
            if len(block_indices) == 1:  # if this is the only block encoded, get its new candidates directly
                latest_block_candidates, latest_block_candidates_errors = self.find_new_candidates(
                    self.b[block_indices[0]], encoding, latest_block_radius)
                # self.a_candidates_per_block.append(latest_block_candidates)
                self.avg_candidates_per_img_history.append(len(latest_block_candidates))
                if len(self.a_candidates):
                    new_candidates_raw = np.array(
                        [np.append(prev_a_candidate, [latest_block_candidate], axis=0) for prev_a_candidate in
                         self.a_candidates for
                         latest_block_candidate in latest_block_candidates])
                    new_candidates_raw_errors = np.array(
                        [prefix_a_candidate_error + latest_block_candidate_error for prefix_a_candidate_error in
                         self.a_candidates_errors for
                         latest_block_candidate_error in latest_block_candidates_errors])
                    if self.cfg.pruning_success_rate != 1.0:
                        pruned_candidates, pruned_candidates_errors = self.prune_candidates(new_candidates_raw,
                                                                                            new_candidates_raw_errors,
                                                                                            latest_block_index)
                        self.a_candidates = pruned_candidates
                        self.a_candidates_errors = pruned_candidates_errors
                    else:
                        self.a_candidates = new_candidates_raw
                        self.a_candidates_errors = new_candidates_raw_errors
                else:
                    self.a_candidates = np.array(
                        [[latest_block_candidate] for latest_block_candidate in latest_block_candidates])
                    self.a_candidates_errors = np.array(latest_block_candidates_errors)
            else:  # else, find all its candidates given the possibilities for encoding of other blocks
                if encoding.format == LinearCodeFormat.MATRIX:
                    prefix_encoding_matrix = encoding.encoding_matrix[:-1]
                elif encoding.format == LinearCodeFormat.AFFINE_SUBSPACE:
                    prefix_encoding_matrix = encoding.prefix_encoding_matrix
                else:
                    pass

                encoded_a_prefix_list = set()
                encoding_to_cur_candidates = {}
                encoding_to_cur_candidates_errors = {}
                for cur_candidate, cur_candidate_error in zip(self.a_candidates, self.a_candidates_errors):
                    cur_candidate_relevant_part = np.take(cur_candidate, block_indices[:-1], axis=0)
                    cur_candidate_encoding = self.encoder.encode_multi_block(prefix_encoding_matrix,
                                                                             cur_candidate_relevant_part)
                    encoded_a_prefix_list.add(tuple(cur_candidate_encoding))
                    encoding_to_cur_candidates.setdefault(tuple(cur_candidate_encoding), []).append(cur_candidate)
                    encoding_to_cur_candidates_errors.setdefault(tuple(cur_candidate_encoding), []).append(
                        cur_candidate_error)
                self.candidates_buckets_num_history.append(len(encoded_a_prefix_list))
                # print(encoded_a_prefix_list)

                latest_block_candidates_per_prefix_encoding, latest_block_candidates_errors_per_prefix_encoding = self.find_new_candidates_multi(
                    self.b[block_indices[-1]], encoding, encoded_a_prefix_list, latest_block_radius)

                # print(latest_block_candidates_per_prefix_encoding)

                self.avg_candidates_per_img_history.append(0.0 if len(latest_block_candidates_per_prefix_encoding) == 0 else
                    sum([len(candidates) for candidates in latest_block_candidates_per_prefix_encoding.values()]) / len(
                        latest_block_candidates_per_prefix_encoding))

                new_candidates_raw = np.array(
                    [np.append(cur_candidate, [latest_block_candidate], axis=0) for cur_prefix_encoding
                     in encoded_a_prefix_list if
                     tuple(cur_prefix_encoding) in latest_block_candidates_errors_per_prefix_encoding
                     for cur_candidate in encoding_to_cur_candidates[tuple(cur_prefix_encoding)]
                     for latest_block_candidate in
                     latest_block_candidates_per_prefix_encoding[tuple(cur_prefix_encoding)]])
                new_candidates_raw_errors = np.array(
                    [cur_candidate_error + latest_block_candidate_error for cur_prefix_encoding in
                     encoded_a_prefix_list if
                     tuple(cur_prefix_encoding) in latest_block_candidates_errors_per_prefix_encoding
                     for cur_candidate_error in encoding_to_cur_candidates_errors[tuple(cur_prefix_encoding)]
                     for latest_block_candidate_error in
                     latest_block_candidates_errors_per_prefix_encoding[tuple(cur_prefix_encoding)]])

                # print(new_candidates_raw)
                if self.cfg.pruning_success_rate != 1.0:
                    pruned_candidates, pruned_candidates_errors = self.prune_candidates(new_candidates_raw,
                                                                                        new_candidates_raw_errors,
                                                                                        latest_block_index)
                    self.a_candidates = pruned_candidates
                    self.a_candidates_errors = pruned_candidates_errors
                else:
                    self.a_candidates = new_candidates_raw
                    self.a_candidates_errors = new_candidates_raw_errors

        else:  # else, just reduce the current candidates
            if encoding.format != LinearCodeFormat.MATRIX:
                raise "Bad encoding format when no new block encoded"
            reduced_candidates = []
            reduced_candidates_errors = []
            for cur_candidate, cur_candidate_error in zip(self.a_candidates, self.a_candidates_errors):
                cur_candidate_relevant_part = np.take(cur_candidate, block_indices, axis=0)
                if self.encoder.is_multi_block_solution(encoding.encoding_matrix, cur_candidate_relevant_part, encoding.encoded_vector):
                    reduced_candidates.append(cur_candidate)
                    reduced_candidates_errors.append(cur_candidate_error)
            self.a_candidates = np.array(reduced_candidates)
            self.a_candidates_errors = np.array(reduced_candidates_errors)

        self.a_candidates_per_block = [
            np.unique([a_block_candidates[i] for a_block_candidates in self.a_candidates], axis=0).tolist() for
            i
            in range(num_blocks_after)]

        # a_candidates_per_err_cnt = {}
        # for a_candidate in self.a_candidates:
        #     a_candidates_per_err_cnt.setdefault(
        #         util.closeness_single_block(self.b[:num_blocks_after][0], a_candidate[0]), []).append(
        #         a_candidate)
        # self.a_candidates_per_err_cnt = a_candidates_per_err_cnt
        # self.min_candidate_error = min([util.closeness_multi_block(self.b, a_candidate, False) for a_candidate in self.a_candidates] or [0])
        # return len(self.a_candidates), [len(a_block_candidates) for a_block_candidates in self.a_candidates_per_block], self.min_candidate_error
        return len(self.a_candidates), [len(a_block_candidates) for a_block_candidates in self.a_candidates_per_block]

    def find_new_candidates(self, block, encoding, radius):
        new_a_candidates = []
        new_a_candidates_errors = []

        if encoding.format == LinearCodeFormat.MATRIX:
            for cur_candidate, cur_error in self.candidates_with_errors_range(block, radius):
                if self.encoder.is_single_block_solution(encoding.encoding_matrix[0], cur_candidate, encoding.encoded_vector):
                    new_a_candidates.append(cur_candidate.tolist())
                    new_a_candidates_errors.append(cur_error)

        elif encoding.format == LinearCodeFormat.AFFINE_SUBSPACE:
            # for cur_candidate in self.single_block_solution_space(encoding.kernel_base, encoding.s0):
            #     cur_error = util.closeness_single_block(block, cur_candidate)
            #     if cur_error <= radius:
            #         new_a_candidates.append(cur_candidate.tolist())
            #         new_a_candidates_errors.append(cur_error)
            kernel_space = np.array([np.matmul(coefficients, encoding.kernel_base) % self.cfg.q for coefficients in product(list(range(self.cfg.q)), repeat=len(encoding.kernel_base))])
            diff = (block - encoding.s0) % self.cfg.q
            def errors(kernel_vector):
                return sum(kernel_vector == diff)
            kernel_space_errors = np.apply_along_axis(errors, 1, kernel_space)
            solutions_mask = (kernel_space_errors <= radius)
            new_a_candidates = (kernel_space[solutions_mask] + encoding.s0) % self.cfg.q
            new_a_candidates_errors = kernel_space_errors[solutions_mask]

        else:
            pass

        return new_a_candidates, new_a_candidates_errors

    def find_new_candidates_multi(self, block, encoding, encoded_a_prefix_list, radius):
        new_candidates_per_prefix_encoding = {}
        new_candidates_errors_per_prefix_encoding = {}

        if encoding.format == LinearCodeFormat.MATRIX:
            # missing_encoding_delta_list = [self.encoder.get_missing_encoding_delta(encoding.encoded_vector, cur_encoding) for
            #                                cur_encoding
            #                                in encoded_a_prefix_list]
            missing_encoding_delta_set = {tuple(self.encoder.get_missing_encoding_delta(encoding.encoded_vector, cur_encoding)) for
                                           cur_encoding
                                           in encoded_a_prefix_list}
            for cur_candidate, cur_error in self.candidates_with_errors_range(block, radius):
                cur_candidate_encoding = self.encoder.encode_single_block(encoding.encoding_matrix[-1], cur_candidate)
                # if any(np.array_equal(cur_candidate_encoding, missing_encoding_delta) for missing_encoding_delta in missing_encoding_delta_list):
                if tuple(cur_candidate_encoding) in missing_encoding_delta_set:
                    prefix_encoding = self.encoder.get_missing_encoding_delta(encoding.encoded_vector, cur_candidate_encoding)
                    new_candidates_per_prefix_encoding.setdefault(tuple(prefix_encoding), []).append(cur_candidate.tolist())
                    new_candidates_errors_per_prefix_encoding.setdefault(tuple(prefix_encoding), []).append(cur_error)
            return new_candidates_per_prefix_encoding, new_candidates_errors_per_prefix_encoding

        elif encoding.format == LinearCodeFormat.AFFINE_SUBSPACE:
            # # errors = []
            # for prefix_encoding in encoded_a_prefix_list:
            #     base_solution_raw = encoding.s0 - np.matmul(prefix_encoding, encoding.image_base_source)
            #     for cur_candidate in self.single_block_solution_space(encoding.kernel_base, base_solution_raw):
            #         cur_error = util.closeness_single_block(block, cur_candidate)
            #         # errors.append(cur_error)
            #         if cur_error <= radius:
            #             new_candidates_per_prefix_encoding.setdefault(tuple(prefix_encoding), []).append(
            #                 cur_candidate.tolist())
            #             new_candidates_errors_per_prefix_encoding.setdefault(tuple(prefix_encoding), []).append(
            #                 cur_error)
            # # print(errors)

            kernel_space = np.array([np.matmul(coefficients, encoding.kernel_base) % self.cfg.q for coefficients in product(list(range(self.cfg.q)), repeat=len(encoding.kernel_base))])
            for prefix_encoding in encoded_a_prefix_list:
                base_solution_raw = encoding.s0 - np.matmul(prefix_encoding, encoding.image_base_source)
                diff = (block - base_solution_raw) % self.cfg.q
                def errors(kernel_vector):
                    return sum(kernel_vector == diff)
                kernel_space_errors = np.apply_along_axis(errors, 1, kernel_space)
                solutions_mask = (kernel_space_errors <= radius)
                new_candidates_per_prefix_encoding[tuple(prefix_encoding)] = (kernel_space[solutions_mask] + base_solution_raw) % self.cfg.q
                new_candidates_errors_per_prefix_encoding[tuple(prefix_encoding)] = kernel_space_errors[solutions_mask]
            return new_candidates_per_prefix_encoding, new_candidates_errors_per_prefix_encoding

        else:
            pass

    def candidates_with_errors_range(self, b, max_num_errors):
        """
        Generate candidates for A given B, with given range of errors.
        """
        for n_err in range(max_num_errors + 1):
            for candidate in self.candidates_with_errors(b, n_err):
                yield candidate, n_err

    def candidates_with_errors(self, b, num_errors):
        """
        Generate candidates for A given B, with given number of errors.
        """
        deltas = product(list(range(1, self.cfg.q)), repeat=self.cfg.block_length - num_errors)
        for cur_delta in deltas:
            for err_indices in itertools.combinations_with_replacement(range(self.cfg.block_length - (num_errors - 1)),
                                                                       num_errors):
                cur_delta_with_err = np.insert(arr=cur_delta, obj=err_indices, values=0)
                cur_candidate = np.mod(np.add(b, cur_delta_with_err), [self.cfg.q] * self.cfg.block_length).astype(int)
                yield cur_candidate

    def single_block_solution_space(self, kernel_base, base_solution_raw):
        for coefficients in product(list(range(self.cfg.q)), repeat=len(kernel_base)):
            yield np.mod(base_solution_raw + np.matmul(coefficients, kernel_base), self.cfg.q)

    def prune_candidates(self, new_candidates_raw, new_candidates_raw_errors, latest_block_index):
        if len(new_candidates_raw) == 0:
            return np.array([]), np.array([])

        if self.cfg.pruning_strategy == PruningStrategy.radii_probabilities:
            total_radius = self.cfg.prefix_radii[latest_block_index]
            pruned_candidates_indices = np.where(new_candidates_raw_errors <= total_radius)[0]
            pruning_rate = 1 - len(pruned_candidates_indices) / len(new_candidates_raw)
            self.pruning_rate_history.append(pruning_rate)
            # print(pruning_rate)
            return new_candidates_raw[pruned_candidates_indices], new_candidates_raw_errors[pruned_candidates_indices]

        elif self.cfg.pruning_strategy == PruningStrategy.relative_weights:
            use_log = True

            a_candidates_per_err_cnt = {}
            for i, err in enumerate(new_candidates_raw_errors):
                a_candidates_per_err_cnt.setdefault(err, []).append(i)
            if use_log:
                if self.cfg.p_err == 0.0:
                    probs_per_err_cnt = {0: 0.0}
                else:
                    probs_per_err_cnt = {err: err * math.log(self.cfg.p_err) + (
                                len(new_candidates_raw[0]) * self.cfg.block_length - err) * math.log(
                        (1 - self.cfg.p_err) / (self.cfg.q - 1))
                                         for err in a_candidates_per_err_cnt.keys()}
                total_prob = scipy.special.logsumexp(
                    [prob + math.log(len(a_candidates_per_err_cnt[err])) for err, prob in probs_per_err_cnt.items()])
            else:
                probs_per_err_cnt = {err: (self.cfg.p_err ** err) * (((1 - self.cfg.p_err) / (self.cfg.q - 1)) ** (len(new_candidates_raw[0]) * self.cfg.block_length - err))
                                     for err in a_candidates_per_err_cnt.keys()}
                total_prob = sum([prob * len(a_candidates_per_err_cnt[err]) for err, prob in probs_per_err_cnt.items()])

            remaining_success_prob = self.cfg.pruning_success_rate / self.cur_success_prob
            cur_block_success_prob = remaining_success_prob ** (1 / (self.cfg.num_blocks - latest_block_index))
            cur_block_fail_prob = 1 - cur_block_success_prob
            if cur_block_fail_prob == 0.0:
                self.pruning_fail_prob_history.append(cur_block_fail_prob)
                return new_candidates_raw, new_candidates_raw_errors

            if use_log:
                cur_pruned_weight = -math.inf
            else:
                cur_pruned_weight = 0.0
            post_pruning_mask = np.full(len(new_candidates_raw_errors), True)
            for err in sorted(a_candidates_per_err_cnt.keys(), reverse=True):
                if use_log:
                    weight_delta = math.log(len(a_candidates_per_err_cnt[err])) + probs_per_err_cnt[err] - total_prob
                else:
                    weight_delta = len(a_candidates_per_err_cnt[err]) * probs_per_err_cnt[err] / total_prob
                if use_log:
                    should_split = weight_delta > util.logminexp(math.log(cur_block_fail_prob), cur_pruned_weight)
                else:
                    should_split = weight_delta > cur_block_fail_prob - cur_pruned_weight
                if should_split:
                    if use_log:
                        delta_split_part = math.floor(math.exp(math.log(len(a_candidates_per_err_cnt[err])) + util.logminexp(math.log(cur_block_fail_prob), cur_pruned_weight) - weight_delta))
                        if delta_split_part != 0:
                            cur_pruned_weight = scipy.special.logsumexp([cur_pruned_weight, math.log(delta_split_part) + probs_per_err_cnt[err] - total_prob])
                    else:
                        delta_split_part = math.floor(len(a_candidates_per_err_cnt[err]) * (cur_block_fail_prob - cur_pruned_weight) / weight_delta)
                        cur_pruned_weight += delta_split_part * probs_per_err_cnt[err] / total_prob
                    split_indices = random.sample(a_candidates_per_err_cnt[err], delta_split_part)
                    post_pruning_mask[split_indices] = False
                    # if use_log:
                    #     cur_pruned_weight = scipy.special.logsumexp([cur_pruned_weight, math.log(delta_split_part) + probs_per_err_cnt[err] - total_prob])
                    # else:
                    #     cur_pruned_weight += delta_split_part * probs_per_err_cnt[err] / total_prob
                    break
                else:
                    post_pruning_mask[a_candidates_per_err_cnt[err]] = False
                    if use_log:
                        cur_pruned_weight = scipy.special.logsumexp([cur_pruned_weight, weight_delta])
                    else:
                        cur_pruned_weight += weight_delta
            # print(1-per_block_fail_prob)
            # print(sum([probs_per_err_cnt[new_candidates_raw_errors[i]] for i, flag in enumerate(post_pruning_mask) if flag]) / total_prob)
            if use_log:
                self.pruning_fail_prob_history.append(math.exp(cur_pruned_weight))
                cur_block_success_prob = 1-math.exp(cur_pruned_weight)
            else:
                self.pruning_fail_prob_history.append(cur_pruned_weight)
                cur_block_success_prob = 1-cur_pruned_weight
            self.cur_success_prob = self.cur_success_prob * cur_block_success_prob
            pruning_rate = 1 - sum(post_pruning_mask) / (len(new_candidates_raw) or 1)
            self.pruning_rate_history.append(pruning_rate)
            # print(pruning_rate)
            return new_candidates_raw[post_pruning_mask], new_candidates_raw_errors[post_pruning_mask]

        else:
            pass

# cfg = ProtocolConfigs(m=3, n=3, num_blocks=10, p_err=0.01, goal_p_bad=0.03)
# b = [0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1, 0, 1, 2]
# bob = Bob(cfg, b)
# candidate = [[2, 0, 1], [0, 1, 1], [1, 2, 2], [2, 0, 2], [2, 0, 1], [2, 0, 1], [1, 1, 1], [1, 0, 1], [2, 0, 2], [2, 0, 1]]
# print(cfg.is_within_radius_multi_block(bob.b, candidate))


# p_err = 0.01
# num_blocks = 20
# block_length = 5
# key_length = block_length * num_blocks
# sample_size = 1000
# for goal_p_bad in np.linspace(0.00001,1.0,10001):
#     print("goal_p_bad = " + str(goal_p_bad))
#     goal_p_bad = goal_p_bad/2 # alter the threshold by a bit to get the desired value in practice
#     # goal_p_bad = max(goal_p_bad - 0.05, 0.0) # alter the threshold by a bit to get the desired value in practice
#     cfg = ProtocolConfigs(m=3, block_length=block_length, num_blocks=num_blocks, p_err=p_err, goal_p_bad=goal_p_bad)
#     # print("overall radius = " + str(cfg.multi_block_radius[num_blocks-1]))
#     close_enough_candidates = 0
#     for k in range(1, sample_size+1):
#         key_generator = KeyGenerator(m=3, p_err=p_err, key_length=key_length)
#         a, b = key_generator.generate_keys()
#         bob = Bob(cfg, b)
#         # close_enough_candidates += cfg.is_within_radius_multi_block(np.array_split(a, num_blocks), np.array_split(b, num_blocks))
#         close_enough_candidates += cfg.is_within_radius_on_all_prefixes(np.array_split(a, num_blocks), np.array_split(b, num_blocks))
#         # if not cfg.is_within_radius_multi_block(np.array_split(a, num_blocks), np.array_split(b, num_blocks)):
#         #     print(key_length - util.hamming_multi_block(a, b))
#     print("actual p_bad = " + str(1 - (close_enough_candidates/sample_size)))
#     print("--------------------")
