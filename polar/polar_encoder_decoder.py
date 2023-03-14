import itertools
from timeit import default_timer as timer
from enum import Enum
import numpy as np

from polar.polar_cfg import IndexType


class ProbResult(Enum):
    SuccessActualIsMax = 0
    SuccessActualSmallerThanMax = 1
    FailActualLargerThanMax = 2
    FailActualIsMax = 3
    FailActualWithinRange = 4
    FailActualSmallerThanMin = 5

class PolarEncoderDecoder:
    def __init__(self, cfg, use_log=False):
        self.cfg = cfg

        self.prob_list = None
        self.actual_prob = None
        self.cfg.use_log = use_log

    def encode(self, x_vec_dist, information_vec):
        u_index = 0
        information_vec_index = 0
        assert (len(x_vec_dist) == self.cfg.N)
        assert (len(information_vec) == self.cfg.num_info_indices)

        (encoded_vector, next_u_index, next_information_vec_index) = self.recursive_encode_decode(information_vec, u_index,
                                                                                                  information_vec_index,
                                                                                                  x_vec_dist)

        assert (next_u_index == len(encoded_vector) == len(x_vec_dist) == self.cfg.N)
        assert (next_information_vec_index == len(information_vec) == self.cfg.num_info_indices)
        return encoded_vector

    def decode(self, x_vec_dist, xy_vec_dist):
        u_index = 0
        info_vec_index = 0

        info_vec = np.full(self.cfg.num_info_indices, -1, dtype=np.int64)

        assert (len(x_vec_dist) == len(xy_vec_dist) == self.cfg.N)

        (encoded_vector, next_u_index, next_info_vec_index) = self.recursive_encode_decode(info_vec, u_index,
                                                                                                  info_vec_index,
                                                                                                  x_vec_dist,
                                                                                                  xy_vec_dist)

        assert (next_u_index == len(encoded_vector) == self.cfg.N)
        assert (next_info_vec_index == len(info_vec) == self.cfg.num_info_indices)

        return info_vec

    def list_decode(self, xy_vec_dist, frozen_values, check_matrix, check_value, actual_info_vec=None):
        u_index = 0
        info_vec_index = 0

        info_vec_list = np.full((self.cfg.scl_l * self.cfg.q, self.cfg.num_info_indices), -1, dtype=np.int64)
        frozen_values_iterator = None
        if len(frozen_values):
            frozen_values_iterator = np.nditer(frozen_values, flags=['f_index'])

        assert (len(xy_vec_dist) == self.cfg.N)

        self.actual_info_vec = actual_info_vec
        if self.cfg.use_log:
            self.actual_prob = 0.0
            self.prob_list = np.array([0.0])
        else:
            self.actual_prob = 1.0
            self.prob_list = np.array([1.0])

        start = timer()
        (info_vec_list, encoded_vector_list, next_u_index, next_information_vector_index, final_list_size,
         original_indices_map, actual_encoding) = self.recursive_list_decode(info_vec_list, u_index, info_vec_index,
                                                                             [xy_vec_dist], frozen_values_iterator, in_list_size=1, actual_xy_vec_dist=xy_vec_dist)
        end = timer()

        assert (1 <= final_list_size <= self.cfg.scl_l)
        assert (len(encoded_vector_list) == final_list_size)
        assert (next_u_index == len(encoded_vector_list[0]) == self.cfg.N)
        assert (next_information_vector_index == self.cfg.num_info_indices)
        assert (len(original_indices_map) == final_list_size)
        assert (np.count_nonzero(original_indices_map) == 0)

        if actual_info_vec is not None:
            explicit_probs, normalization = normalize([self.calc_explicit_prob(information, frozen_values, xy_vec_dist) for information
                                                       in info_vec_list[:self.cfg.scl_l]], use_log=self.cfg.use_log)
            actual_explicit_prob = self.calc_explicit_prob(actual_info_vec, frozen_values, xy_vec_dist)
            if self.cfg.use_log:
                actual_explicit_prob = actual_explicit_prob - normalization
            else:
                actual_explicit_prob = actual_explicit_prob / normalization

            for i, information in enumerate(info_vec_list[:self.cfg.scl_l]):
                if np.array_equal(information, actual_info_vec):
                    max_prob = max(self.prob_list)
                    if self.prob_list[i] == max_prob:
                        prob_result = ProbResult.SuccessActualIsMax
                    else:
                        prob_result = ProbResult.SuccessActualSmallerThanMax
                    return information, prob_result

            max_prob = max(self.prob_list)
            if self.actual_prob > max_prob:
                prob_result = ProbResult.FailActualLargerThanMax
            elif self.actual_prob == max_prob:
                prob_result = ProbResult.FailActualIsMax
            elif self.actual_prob >= min(self.prob_list):
                prob_result = ProbResult.FailActualWithinRange
            else:
                prob_result = ProbResult.FailActualSmallerThanMin
            return info_vec_list[0], prob_result

        candidate_list = np.array([np.array_equal(np.matmul(info_vec, check_matrix) % self.cfg.q, check_value) for info_vec in info_vec_list[:self.cfg.scl_l]])
        if True in candidate_list:
            for i, val in enumerate(candidate_list):
                if val:
                    return info_vec_list[i], None

        return info_vec_list[0], None

    def calc_explicit_prob(self, information, frozen_vec, xy_vec_dist):
        guess = polar_transform(self.cfg.q, self.merge_info_frozen(information, frozen_vec))
        probs_list = [xy_prob[guess[i]] for i, xy_prob in enumerate(xy_vec_dist.probs)]
        if self.cfg.use_log:
            guess_prob = sum(probs_list)
        else:
            guess_prob = np.prod(probs_list)
        return guess_prob

    def merge_info_frozen(self, actual_info_vec, frozen_vec):
        merged_vec = np.empty(self.cfg.N, dtype=np.int)
        merged_vec[list(self.cfg.info_set)] = actual_info_vec
        merged_vec[list(self.cfg.frozen_set)] = frozen_vec
        return merged_vec

    def recursive_encode_decode(self, info_vec, u_index, info_vec_index, x_vec_dist,
                                xy_vec_dist=None):
        # By default, we assume encoding, and add small corrections for decoding.
        encoded_vec = np.full(len(x_vec_dist), -1, dtype=np.int64)
        decoding = xy_vec_dist is not None

        if len(x_vec_dist) == 1:
            if self.cfg.index_types[u_index] == IndexType.info:
                if decoding:
                    marginalized_vec = xy_vec_dist.calc_marginalized_probs()
                    info_vec[info_vec_index] = np.argmax(marginalized_vec)
                encoded_vec[0] = info_vec[info_vec_index]
                next_u_index = u_index + 1
                next_info_vec_index = info_vec_index + 1
            else:
                # marginalized_vec = xVectorDistribution.calc_marginalized_probs()
                # encoded_vec[0] = min(
                #     np.searchsorted(np.cumsum(marginalized_vec), self.randomlyGeneratedNumbers[uIndex]),
                #     self.cfg.q - 1)
                encoded_vec[0] = 0
                next_u_index = u_index + 1
                next_info_vec_index = info_vec_index

            return (encoded_vec, next_u_index, next_info_vec_index)
        else:
            x_minus_vec_dist = x_vec_dist.minus_transform()
            x_minus_vec_dist.normalize()
            if decoding:
                xy_minus_vec_dist = xy_vec_dist.minus_transform()
                xy_minus_vec_dist.normalize()
            else:
                xy_minus_vec_dist = None

            (minus_encoded_vec, next_u_index, next_info_vec_index) = self.recursive_encode_decode(info_vec,
                                                                                                          u_index,
                                                                                                          info_vec_index,
                                                                                                          x_minus_vec_dist,
                                                                                                          xy_minus_vec_dist)

            x_plus_vec_dist = x_vec_dist.plus_transform(minus_encoded_vec)
            x_plus_vec_dist.normalize()
            if decoding:
                xy_plus_vec_dist = xy_vec_dist.plus_transform(minus_encoded_vec)
                xy_plus_vec_dist.normalize()
            else:
                xy_plus_vec_dist = None

            u_index = next_u_index
            info_vec_index = next_info_vec_index
            (plus_encoded_vec, next_u_index, next_info_vec_index) = self.recursive_encode_decode(info_vec,
                                                                                                         u_index,
                                                                                                         info_vec_index,
                                                                                                         x_plus_vec_dist,
                                                                                                         xy_plus_vec_dist)

            half_N = len(x_vec_dist) // 2

            for half_i in range(half_N):
                encoded_vec[2 * half_i] = (minus_encoded_vec[half_i] + plus_encoded_vec[half_i]) % self.cfg.q
                encoded_vec[2 * half_i + 1] = (-plus_encoded_vec[half_i] + self.cfg.q) % self.cfg.q

            return (encoded_vec, next_u_index, next_info_vec_index)

    def recursive_list_decode(self, info_vec_list, u_index, info_vec_index,
                              xy_vec_dist_list, frozen_vec_iterator=None, in_list_size=1,
                              actual_xy_vec_dist=None):
        assert (in_list_size <= self.cfg.scl_l)
        assert (in_list_size == len(self.prob_list))
        segment_size = len(xy_vec_dist_list[0])

        if segment_size == 1:
            if self.cfg.index_types[u_index] == IndexType.info:
                encoded_vec_list = np.full((in_list_size * self.cfg.q, segment_size), -1, dtype=np.int64)
                new_prob_list = np.empty(in_list_size * self.cfg.q, dtype=np.float)
                for i in range(in_list_size):
                    info_vec = info_vec_list[i]
                    marginalized_vec = xy_vec_dist_list[i].calc_marginalized_probs()
                    start = timer()
                    for s in range(self.cfg.q):
                        if s > 0:
                            info_vec_list[s * in_list_size + i] = info_vec  # branch the paths q times
                        info_vec_list[s * in_list_size + i][info_vec_index] = s
                        encoded_vec_list[s * in_list_size + i][0] = s
                        if self.cfg.use_log:
                            new_prob_list[s * in_list_size + i] = self.prob_list[i] + marginalized_vec[s]
                        else:
                            new_prob_list[s * in_list_size + i] = self.prob_list[i] * marginalized_vec[s]
                    end = timer()
                    self.info_time += end - start
                new_list_size = in_list_size * self.cfg.q
                next_u_index = u_index + 1
                next_info_vec_index = info_vec_index + 1

                if new_list_size > self.cfg.scl_l:
                    if self.cfg.use_log:
                        new_list_size = min(new_list_size - np.isneginf(new_prob_list).sum(), self.cfg.scl_l)
                    else:
                        new_list_size = min(np.count_nonzero(new_prob_list), self.cfg.scl_l)
                    indices_to_keep = np.argpartition(new_prob_list, -new_list_size)[-new_list_size:]
                    orig_indices_map = indices_to_keep % in_list_size
                    start = timer()
                    info_vec_list[0:new_list_size] = info_vec_list[indices_to_keep]
                    info_vec_list[new_list_size:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encoded_vec_list = encoded_vec_list[indices_to_keep]
                    new_prob_list = new_prob_list[indices_to_keep]
                else:
                    orig_indices_map = np.tile(np.arange(in_list_size), self.cfg.q)

                self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)

                if actual_xy_vec_dist is not None:
                    actual_encoded_vec = [self.actual_info_vec[info_vec_index]]
                    if self.cfg.use_log:
                        self.actual_prob += actual_xy_vec_dist.calc_marginalized_probs()[actual_encoded_vec[0]] - norm_wt
                    else:
                        self.actual_prob *= actual_xy_vec_dist.calc_marginalized_probs()[actual_encoded_vec[0]] / norm_wt
            else:
                new_list_size = in_list_size
                frozen_val = frozen_vec_iterator[0]
                encoded_vec_list = np.full((in_list_size, segment_size), frozen_val, dtype=np.int64)
                if actual_xy_vec_dist is not None:
                    actual_encoded_vec = [frozen_val]
                frozen_vec_iterator.iternext()
                # encoded_vec_list = np.full((inListSize, segment_size), 0, dtype=np.int64)
                next_u_index = u_index + 1
                next_info_vec_index = info_vec_index
                orig_indices_map = np.arange(in_list_size)
                if self.cfg.use_log:
                    self.prob_list, norm_wt = normalize([prob + xy_vec_dist_list[i].calc_marginalized_probs()[frozen_val] for i, prob in enumerate(self.prob_list)], use_log=self.cfg.use_log)
                    self.actual_prob += actual_xy_vec_dist.calc_marginalized_probs()[frozen_val] - norm_wt
                else:
                    self.prob_list, norm_wt = normalize([prob * xy_vec_dist_list[i].calc_marginalized_probs()[frozen_val] for i, prob in enumerate(self.prob_list)], use_log=self.cfg.use_log)
                    self.actual_prob *= actual_xy_vec_dist.calc_marginalized_probs()[frozen_val] / norm_wt

            return (info_vec_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                    orig_indices_map, actual_encoded_vec)
        else:
            num_info_symbols = np.sum((self.cfg.index_types[u_index:u_index + segment_size] == IndexType.info))

            # Rate-0 node
            if num_info_symbols == 0:
                frozen_vec = np.empty(segment_size, dtype=np.int64)
                for i in range(segment_size):
                    frozen_vec[i] = frozen_vec_iterator[0]
                    frozen_vec_iterator.iternext()
                encoded_vec = polar_transform(self.cfg.q, frozen_vec)
                encoded_vec_list = [encoded_vec] * in_list_size
                if self.cfg.use_log:
                    new_prob_list = [prob + np.sum([xy_vec_dist.probs[i, encoded_vec[i]] for i in range(segment_size)]) for xy_vec_dist, prob in zip(xy_vec_dist_list, self.prob_list)]
                    self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)
                    self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, encoded_vec[i]] for i in range(segment_size)]) - norm_wt
                else:
                    new_prob_list = [prob * np.product([xy_vec_dist.probs[i, encoded_vec[i]] for i in range(segment_size)]) for xy_vec_dist, prob in zip(xy_vec_dist_list, self.prob_list)]
                    self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)
                    self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, encoded_vec[i]] for i in
                                                    range(segment_size)]) / norm_wt

                next_u_index = u_index + segment_size
                next_info_vec_index = info_vec_index
                new_list_size = in_list_size
                orig_indices_map = np.arange(in_list_size)
                actual_encoded_vec = encoded_vec
                return (info_vec_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                        orig_indices_map, actual_encoded_vec)

            # Rep node
            if num_info_symbols == 1:
                k = np.where(self.cfg.index_types[u_index:u_index + segment_size] == IndexType.info)[0][0]
                input_vec_splits = np.empty((self.cfg.q, segment_size), dtype=np.int64)
                for i in range(segment_size):
                    if i != k:
                        input_vec_splits[:, i] = frozen_vec_iterator[0]
                        frozen_vec_iterator.iternext()
                    else:
                        input_vec_splits[:, i] = np.arange(self.cfg.q)
                encoded_vec_splits = np.array([polar_transform(self.cfg.q, v) for v in input_vec_splits])

                new_prob_list = np.empty(in_list_size * self.cfg.q, dtype=np.float)
                for i in range(in_list_size):
                    info_vec = info_vec_list[i]
                    start = timer()
                    for s in range(self.cfg.q):
                        if s > 0:
                            info_vec_list[s * in_list_size + i] = info_vec  # branch the paths q times
                        info_vec_list[s * in_list_size + i][info_vec_index] = s
                        if self.cfg.use_log:
                            new_prob_list[s * in_list_size + i] = self.prob_list[i] + np.sum([xy_vec_dist_list[i].probs[j, encoded_vec_splits[s, j]] for j in range(segment_size)])
                        else:
                            new_prob_list[s * in_list_size + i] = self.prob_list[i] * np.product([xy_vec_dist_list[i].probs[j, encoded_vec_splits[s, j]] for j in range(segment_size)])
                    end = timer()
                    self.info_time += end - start
                new_list_size = in_list_size * self.cfg.q

                if new_list_size > self.cfg.scl_l:
                    if self.cfg.use_log:
                        new_list_size = min(new_list_size - np.isneginf(new_prob_list).sum(), self.cfg.scl_l)
                    else:
                        new_list_size = min(np.count_nonzero(new_prob_list), self.cfg.scl_l)
                    indices_to_keep = np.argpartition(new_prob_list, -new_list_size)[-new_list_size:]
                    orig_indices_map = indices_to_keep % in_list_size
                    start = timer()
                    info_vec_list[0:new_list_size] = info_vec_list[indices_to_keep]
                    info_vec_list[new_list_size:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encoded_vec_list = encoded_vec_splits[indices_to_keep // in_list_size]
                    new_prob_list = new_prob_list[indices_to_keep]
                else:
                    encoded_vec_list = np.repeat(encoded_vec_splits, in_list_size, axis=0)
                    orig_indices_map = np.tile(np.arange(in_list_size), self.cfg.q)

                self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)

                if actual_xy_vec_dist is not None:
                    actual_encoded_vec = encoded_vec_splits[self.actual_info_vec[info_vec_index]]
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in range(segment_size)]) - norm_wt
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in range(segment_size)]) / norm_wt

                next_u_index = u_index + segment_size
                next_info_vec_index = info_vec_index + 1
                return (info_vec_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                        orig_indices_map, actual_encoded_vec)

            # Rate-1 node
            if num_info_symbols == segment_size:
                fork_size = self.cfg.q ** 2
                encoded_vec_list = np.empty((in_list_size * fork_size, segment_size), dtype=np.int64)
                new_prob_list = np.empty(in_list_size * fork_size, dtype=np.float)
                for i in range(in_list_size):
                    # Pick the 2 least reliable indices
                    [j1, j2] = self.pick_least_reliable_indices(xy_vec_dist_list[i].probs, 2)
                    # Fork there
                    encoded_vec_list[fork_size*i : fork_size*(i + 1)], new_prob_list[fork_size*i : fork_size*(i + 1)] = self.fork_indices(self.prob_list[i], xy_vec_dist_list[i].probs, segment_size, [j1, j2])
                new_list_size = in_list_size * fork_size

                # Prune
                if new_list_size > self.cfg.scl_l:
                    if self.cfg.use_log:
                        new_list_size = min(new_list_size - np.isneginf(new_prob_list).sum(), self.cfg.scl_l)
                    else:
                        new_list_size = min(np.count_nonzero(new_prob_list), self.cfg.scl_l)
                    indices_to_keep = np.argpartition(new_prob_list, -new_list_size)[-new_list_size:]
                    encoded_vec_list = encoded_vec_list[indices_to_keep]
                    new_prob_list = new_prob_list[indices_to_keep]
                    orig_indices_map = indices_to_keep // fork_size
                else:
                    orig_indices_map = np.repeat(np.arange(in_list_size), fork_size)

                # Normalize
                self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)
                if actual_xy_vec_dist is not None:
                    actual_encoded_vec = polar_transform(self.cfg.q, self.actual_info_vec[info_vec_index:info_vec_index + segment_size])
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in
                                                    range(segment_size)]) - norm_wt
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in
                                                        range(segment_size)]) / norm_wt

                # Update informationList
                start = timer()
                info_vec_list[0:new_list_size] = info_vec_list[orig_indices_map]
                info_vec_list[0:new_list_size, info_vec_index:info_vec_index + segment_size] = [polar_transform(self.cfg.q, encodedVector) for encodedVector in encoded_vec_list]
                info_vec_list[new_list_size:] = -1
                end = timer()
                self.info_time += end - start

                next_u_index = u_index + segment_size
                next_info_vec_index = info_vec_index + segment_size

                return (info_vec_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                        orig_indices_map, actual_encoded_vec)

            # SPC node
            if num_info_symbols == segment_size - 1:
                num_forked_indices = 3

                fork_size = self.cfg.q ** num_forked_indices
                encoded_vec_list = np.empty((in_list_size * fork_size, segment_size), dtype=np.int64)
                new_prob_list = np.empty(in_list_size * fork_size, dtype=np.float)
                frozen_val = frozen_vec_iterator[0]
                for i in range(in_list_size):
                    # Pick the least reliable indices
                    leastReliableIndices = self.pick_least_reliable_indices(xy_vec_dist_list[i].probs, num_forked_indices + 1)
                    # Fork there
                    encoded_vec_list[fork_size*i: fork_size*(i + 1)], new_prob_list[fork_size*i: fork_size*(i + 1)] = self.fork_indices_spc(self.prob_list[i], xy_vec_dist_list[i].probs, segment_size, leastReliableIndices, frozen_val)
                frozen_vec_iterator.iternext()
                new_list_size = in_list_size * fork_size

                # Prune
                if new_list_size > self.cfg.scl_l:
                    if self.cfg.use_log:
                        new_list_size = min(new_list_size - np.isneginf(new_prob_list).sum(), self.cfg.scl_l)
                    else:
                        new_list_size = min(np.count_nonzero(new_prob_list), self.cfg.scl_l)
                    indices_to_keep = np.argpartition(new_prob_list, -new_list_size)[-new_list_size:]
                    encoded_vec_list = encoded_vec_list[indices_to_keep]
                    new_prob_list = new_prob_list[indices_to_keep]
                    orig_indices_map = indices_to_keep // fork_size
                else:
                    orig_indices_map = np.repeat(np.arange(in_list_size), fork_size)

                # Normalize
                self.prob_list, norm_wt = normalize(new_prob_list, use_log=self.cfg.use_log)
                if actual_xy_vec_dist is not None:
                    actual_encoded_vec = polar_transform(self.cfg.q, np.concatenate(([frozen_val], self.actual_info_vec[info_vec_index:info_vec_index + segment_size - 1]), axis=None))
                    if self.cfg.use_log:
                        self.actual_prob += np.sum([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in
                                                    range(segment_size)]) - norm_wt
                    else:
                        self.actual_prob *= np.product([actual_xy_vec_dist.probs[i, actual_encoded_vec[i]] for i in
                                                        range(segment_size)]) / norm_wt

                # Update informationList
                start = timer()
                info_vec_list[0:new_list_size] = info_vec_list[orig_indices_map]
                info_vec_list[0:new_list_size, info_vec_index:info_vec_index + segment_size - 1] = [polar_transform(self.cfg.q, encodedVector)[1:] for encodedVector in encoded_vec_list]
                info_vec_list[new_list_size:] = -1
                end = timer()
                self.info_time += end - start

                next_u_index = u_index + segment_size
                next_info_vec_index = info_vec_index + segment_size - 1

                return (info_vec_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                        orig_indices_map, actual_encoded_vec)

            start = timer()
            xy_minus_vec_dist_list = []
            for i in range(in_list_size):
                xy_minus_vec_dist = xy_vec_dist_list[i].minus_transform()
                xy_minus_vec_dist.normalize()
                xy_minus_vec_dist_list.append(xy_minus_vec_dist)
            # xy_minus_vec_dist_list, xyMinusNormalizationVector = normalizeDistList(xy_minus_vec_dist_list)
            end = timer()
            self.transform_time += end-start
            if actual_xy_vec_dist is not None:
                actual_xy_minus_vec_dist = actual_xy_vec_dist.minus_transform()
                actual_xy_minus_vec_dist.normalize()
                # actual_xy_minus_vec_dist.normalize(xyMinusNormalizationVector)

            (minus_info_list, minus_encoded_vec_list, next_u_index, next_info_vec_index, minus_list_size,
             minus_orig_indices_map, minus_actual_encoded_vec) = self.recursive_list_decode(info_vec_list, u_index, info_vec_index,
                                                                                             xy_minus_vec_dist_list, frozen_vec_iterator,
                                                                                             in_list_size, actual_xy_vec_dist=actual_xy_minus_vec_dist)

            start = timer()
            xy_plus_vec_dist_list = []
            for i in range(minus_list_size):
                origI = minus_orig_indices_map[i]

                xy_plus_vec_dist = xy_vec_dist_list[origI].plus_transform(minus_encoded_vec_list[i])
                xy_plus_vec_dist.normalize()
                xy_plus_vec_dist_list.append(xy_plus_vec_dist)
            # xy_plus_vec_dist_list, xyPlusNormalizationVector = normalizeDistList(xy_plus_vec_dist_list)
            end = timer()
            self.transform_time += end-start
            if actual_xy_vec_dist is not None:
                actual_xy_plus_vec_dist = actual_xy_vec_dist.plus_transform(minus_actual_encoded_vec)
                actual_xy_plus_vec_dist.normalize()
                # actual_xy_plus_vec_dist.normalize(xyPlusNormalizationVector)

            u_index = next_u_index
            info_vec_index = next_info_vec_index
            (plus_info_list, plus_encoded_vec_list, next_u_index, next_info_vec_index, plus_list_size,
             plus_orig_indices_map, plus_actual_encoded_vec) = self.recursive_list_decode(minus_info_list, u_index, info_vec_index,
                                                                                           xy_plus_vec_dist_list, frozen_vec_iterator,
                                                                                           minus_list_size, actual_xy_vec_dist=actual_xy_plus_vec_dist)

            new_list_size = plus_list_size

            encoded_vec_list = np.full((new_list_size, segment_size), -1, dtype=np.int64)
            # halfLength = segment_size // 2

            start = timer()
            for i in range(new_list_size):
                minus_i = plus_orig_indices_map[i]
                encoded_vec_list[i][::2] = minus_encoded_vec_list[minus_i] + plus_encoded_vec_list[i]
                encoded_vec_list[i][1::2] = -plus_encoded_vec_list[i]
                encoded_vec_list[i] %= self.cfg.q
            end = timer()
            self.encoding_time += end-start

            if actual_xy_vec_dist is not None:
                actual_encoded_vec = np.full(segment_size, -1, dtype=np.int64)
                actual_encoded_vec[::2] = np.add(minus_actual_encoded_vec, plus_actual_encoded_vec)
                actual_encoded_vec[1::2] = np.array(plus_actual_encoded_vec) * (-1)
                actual_encoded_vec %= self.cfg.q

            orig_indices_map = minus_orig_indices_map[plus_orig_indices_map]

            return (plus_info_list, encoded_vec_list, next_u_index, next_info_vec_index, new_list_size,
                    orig_indices_map, actual_encoded_vec)

    def pick_least_reliable_indices(self, xy_dist, num_unreliable_indices):
        scores = [self.reliability(cur_probs) for cur_probs in xy_dist]
        return np.argpartition(scores, range(-num_unreliable_indices, 1))[-num_unreliable_indices:]

    def reliability(self, probs):
        max_probs = np.partition(probs, -2)[-2:]
        if self.cfg.use_log:
            return max_probs[0] - max_probs[1]
        else:
            return max_probs[0] / max_probs[1]

    def fork_indices(self, cur_prob, xy_dist, segment_size, indices_to_fork):
        num_indices_to_fork = len(indices_to_fork)
        if num_indices_to_fork != 2:
            raise "No support for number of indices to fork: " + str(num_indices_to_fork)

        num_forks = self.cfg.q ** num_indices_to_fork
        forks = np.empty((num_forks, segment_size), dtype=np.int64)
        constant_indices_mask = np.ones(segment_size, dtype=bool)
        constant_indices_mask[indices_to_fork] = False

        forks[:, constant_indices_mask] = np.argmax(xy_dist[constant_indices_mask], axis=1)
        forks[:, indices_to_fork] = list(itertools.product(np.arange(self.cfg.q), repeat=num_indices_to_fork))

        fork_prob_combinations = list(itertools.product(xy_dist[indices_to_fork[0]], xy_dist[indices_to_fork[1]]))
        if self.cfg.use_log:
            base_prob = cur_prob + sum(np.max(xy_dist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xy_dist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def fork_indices_spc(self, cur_prob, xy_dist, segment_size, least_reliable_indices, frozen_val):
        num_indices_to_fork = len(least_reliable_indices) - 1
        dependent_index = least_reliable_indices[-1]
        indices_to_fork = least_reliable_indices[:-1]
        # dependent_index = least_reliable_indices[0]
        # indices_to_fork = least_reliable_indices[1:]

        num_forks = self.cfg.q ** num_indices_to_fork
        forks = np.empty((num_forks, segment_size), dtype=np.int64)
        constant_indices_mask = np.ones(segment_size, dtype=bool)
        constant_indices_mask[least_reliable_indices] = False

        forked_values = np.argmax(xy_dist[constant_indices_mask], axis=1)
        forks[:, constant_indices_mask] = forked_values
        base_frozen_delta = (frozen_val - sum(forked_values)) % self.cfg.q
        forks[:, indices_to_fork] = list(itertools.product(np.arange(self.cfg.q), repeat=num_indices_to_fork))
        forks[:, dependent_index] = np.mod(
            (base_frozen_delta - np.sum(forks[:, indices_to_fork], axis=1)), self.cfg.q)

        fork_prob_combinations = np.array([[xy_dist[i, fork[i]] for i in least_reliable_indices] for fork in forks])
        if self.cfg.use_log:
            base_prob = cur_prob + sum(np.max(xy_dist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xy_dist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def calculate_syndrome_and_complement(self, u_message):
        y = polar_transform(self.cfg.q, u_message)

        w = np.copy(y)
        w[list(self.cfg.info_set)] = 0
        w[list(self.cfg.frozen_set)] *= self.cfg.q - 1
        w[list(self.cfg.frozen_set)] %= self.cfg.q

        u = y
        u[list(self.cfg.frozen_set)] = 0

        return w, u

    def get_message_info_bits(self, u_message):
        return u_message[list(self.cfg.info_set)]

    def get_message_frozen_bits(self, u_message):
        return u_message[list(self.cfg.frozen_set)]

def normalize(prob_list, use_log=False):
    max_prob = np.max(prob_list)
    if use_log:
        return prob_list - max_prob, max_prob
    else:
        return prob_list / max_prob, max_prob

def calc_normalization_vec(dist_list):
    segment_size = len(dist_list[0].probs)
    normalization = np.zeros(segment_size)
    for i in range(segment_size):
        normalization[i] = max([dist.probs[i].max(axis=0) for dist in dist_list])
    return normalization

def normalize_dist_list(dist_list):
    normalization_vector = calc_normalization_vec(dist_list)
    for dist in dist_list:
        dist.normalize(normalization_vector)
    return dist_list, normalization_vector

def polar_transform(q, xvec):
    # print("xvec =", xvec)
    if len(xvec) == 1:
        return xvec
    else:
        if len(xvec) % 2 != 0:
            print(xvec)
        assert (len(xvec) % 2 == 0)

        v1 = []
        v2 = []
        for i in range((len(xvec) // 2)):
            v1.append((xvec[2 * i] + xvec[2 * i + 1]) % q)
            v2.append((q - xvec[2 * i + 1]) % q)

        u1 = polar_transform(q, v1)
        u2 = polar_transform(q, v2)

        return np.concatenate((u1, u2))