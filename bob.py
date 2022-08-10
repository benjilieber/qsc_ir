import itertools
import numpy as np
import util

from encoder import Encoder
from itertools import product


class Bob(object):

    def __init__(self, protocol_configs, b):
        self.cfg = protocol_configs
        self.encoder = Encoder(protocol_configs.base, protocol_configs.block_length)
        self.b = np.array_split(b, protocol_configs.num_blocks)  # Bob's private key, partitioned into blocks
        self.b_base_m = np.array(list(map(self.encoder.base_3_to_m, self.b)))  # B in base m
        self.a_candidates = []
        self.a_candidates_errors = []
        self.a_candidates_per_block = []
        # self.a_candidates_per_err_cnt = {}
        # self.min_candidate_error = 0
        self.candidates_buckets_num_history = []
        self.avg_candidates_per_img_history = []
        self.pruning_rate_history = []

    def decode_multi_block(self, block_indices, encoding_matrix, encoded_a):
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
            total_radius = self.cfg.prefix_radii[latest_block_index]
            if len(block_indices) == 1:  # if this is the only block encoded, get its new candidates directly
                latest_block_candidates, latest_block_candidates_errors = self.find_new_candidates(self.b[block_indices[0]], encoding_matrix[0],
                                                                    encoded_a, latest_block_radius)
                # self.a_candidates_per_block.append(latest_block_candidates)
                self.avg_candidates_per_img_history.append(len(latest_block_candidates))
                if self.a_candidates:
                    # new_candidates_raw = [prev_a_candidate + [latest_block_candidate] for i, prev_a_candidate in
                    #                      enumerate(self.a_candidates) for
                    #                      j, latest_block_candidate in enumerate(latest_block_candidates)]
                    # new_candidates = self.prune(new_candidates_raw, latest_block_index)
                    new_candidates = [prev_a_candidate + [latest_block_candidate] for i, prev_a_candidate in
                                         enumerate(self.a_candidates) for
                                         j, latest_block_candidate in enumerate(latest_block_candidates) if self.a_candidates_errors[i] + latest_block_candidates_errors[j] <= total_radius]
                    self.a_candidates_errors = [self.a_candidates_errors[i] + latest_block_candidates_errors[j] for i, prev_a_candidate in
                                         enumerate(self.a_candidates) for
                                         j, latest_block_candidate in enumerate(latest_block_candidates) if self.a_candidates_errors[i] + latest_block_candidates_errors[j] <= total_radius]
                    self.pruning_rate_history.append(1-len(new_candidates)/(len(self.a_candidates) * len(latest_block_candidates)))
                    self.a_candidates = new_candidates
                else:
                    self.a_candidates = [[latest_block_candidate] for latest_block_candidate in latest_block_candidates]
                    self.a_candidates_errors = latest_block_candidates_errors
            else:  # else, find all its candidates given the possibilities for encoding of other blocks
                encodings = set()
                encoding_to_cur_candidates = {}
                encoding_to_cur_candidates_errors = {}
                for cur_candidate, cur_candidate_error in zip(self.a_candidates, self.a_candidates_errors):
                    cur_candidate_relevant_part = np.take(cur_candidate, block_indices[:-1], axis=0)
                    cur_candidate_encoding = self.encoder.encode_multi_block(encoding_matrix[:-1],
                                                                             cur_candidate_relevant_part)
                    encodings.add(tuple(cur_candidate_encoding))
                    encoding_to_cur_candidates.setdefault(tuple(cur_candidate_encoding), []).append(cur_candidate)
                    encoding_to_cur_candidates_errors.setdefault(tuple(cur_candidate_encoding), []).append(cur_candidate_error)
                self.candidates_buckets_num_history.append(len(encodings))
                missing_encoding_delta_list = [self.encoder.get_missing_encoding_delta(encoded_a, cur_encoding) for
                                               cur_encoding
                                               in encodings]
                latest_block_candidates_per_encoding, latest_block_candidates_errors_per_encoding = self.find_new_candidates_multi(self.b[block_indices[-1]],
                                                                                       encoding_matrix[-1],
                                                                                       missing_encoding_delta_list, latest_block_radius)
                self.avg_candidates_per_img_history.append(sum([len(candidates) for candidates in latest_block_candidates_per_encoding.values()]) / len(latest_block_candidates_per_encoding))
                new_candidates = [cur_candidate + [latest_block_candidate] for cur_encoding, missing_encoding_delta in
                                  zip(encodings, missing_encoding_delta_list)
                                  for cur_candidate, cur_candidate_error in zip(encoding_to_cur_candidates[tuple(cur_encoding)], encoding_to_cur_candidates_errors[tuple(cur_encoding)])
                                  for latest_block_candidate, latest_block_candidate_error in
                                  zip(latest_block_candidates_per_encoding.get(tuple(missing_encoding_delta), []), latest_block_candidates_errors_per_encoding.get(tuple(missing_encoding_delta), []))
                                  if cur_candidate_error + latest_block_candidate_error <= total_radius]
                new_candidates_errors = [cur_candidate_error + latest_block_candidate_error for cur_encoding, missing_encoding_delta in
                                  zip(encodings, missing_encoding_delta_list)
                                  for cur_candidate_error in encoding_to_cur_candidates_errors[tuple(cur_encoding)]
                                  for latest_block_candidate_error in
                                  latest_block_candidates_errors_per_encoding.get(tuple(missing_encoding_delta), [])
                                  if cur_candidate_error + latest_block_candidate_error <= total_radius]
                self.pruning_rate_history.append(1 - len(new_candidates_errors)/len([True for cur_encoding, missing_encoding_delta in
                                  zip(encodings, missing_encoding_delta_list)
                                  for _ in encoding_to_cur_candidates_errors[tuple(cur_encoding)]
                                  for _ in
                                  latest_block_candidates_errors_per_encoding.get(tuple(missing_encoding_delta), [])]))

                self.a_candidates = new_candidates
                self.a_candidates_errors = new_candidates_errors

        else:  # else, just reduce the current candidates
            reduced_candidates = []
            reduced_candidates_errors = []
            for cur_candidate, cur_candidate_error in zip(self.a_candidates, self.a_candidates_errors):
                cur_candidate_relevant_part = np.take(cur_candidate, block_indices, axis=0)
                if self.encoder.is_multi_block_solution(encoding_matrix, cur_candidate_relevant_part, encoded_a):
                    reduced_candidates.append(cur_candidate)
                    reduced_candidates_errors.append(cur_candidate_error)
            self.a_candidates = reduced_candidates
            self.a_candidates_errors = reduced_candidates_errors

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

    def find_new_candidates(self, block, encoding_matrix, encoded_a, radius):
        new_a_candidates = []
        new_a_candidates_errors = []
        for cur_candidate, cur_error in self.candidates_with_errors_range(block, radius):
            if self.encoder.is_single_block_solution(encoding_matrix, cur_candidate, encoded_a):
                new_a_candidates.append(cur_candidate.tolist())
                new_a_candidates_errors.append(cur_error)
        return new_a_candidates, new_a_candidates_errors

    def find_new_candidates_multi(self, block, encoding_matrix, encoded_a_list, radius):
        new_candidates_per_encoding = {}
        new_candidates_errors_per_encoding = {}
        for cur_candidate, cur_error in self.candidates_with_errors_range(block, radius):
            cur_candidate_encoding = self.encoder.encode_single_block(encoding_matrix, cur_candidate)
            if any(np.array_equal(cur_candidate_encoding, encoded_a) for encoded_a in encoded_a_list):
                new_candidates_per_encoding.setdefault(tuple(cur_candidate_encoding), []).append(cur_candidate.tolist())
                new_candidates_errors_per_encoding.setdefault(tuple(cur_candidate_encoding), []).append(cur_error)
        return new_candidates_per_encoding, new_candidates_errors_per_encoding

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
        deltas = product([1, 2], repeat=self.cfg.block_length - num_errors)
        for cur_delta in deltas:
            for err_indices in itertools.combinations_with_replacement(range(self.cfg.block_length - (num_errors - 1)),
                                                                       num_errors):
                cur_delta_with_err = np.insert(arr=cur_delta, obj=err_indices, values=0)
                cur_candidate = np.mod(np.add(b, cur_delta_with_err), [3] * self.cfg.block_length).astype(int)
                yield cur_candidate

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
