import itertools

import numpy as np
import math
from itertools import repeat

import scipy

import util
from ldpc.f import F
from q import Q
from r import R
from stats import Stats


class LdpcDecoder(object):

    def __init__(self, base, p_err, encoding_matrix, encoded_a, a=None, use_forking=False, use_hints=False, max_candidates_num=None, success_rate=None):
        self.base = base
        self.p_err = p_err
        self.encoded_a = encoded_a
        assert(not (use_forking and use_hints))
        self.use_forking = use_forking
        self.use_hints = use_hints
        self.encoding_matrix = encoding_matrix
        self.num_noise_symbols = len(encoding_matrix.indptr_r) - 1
        self.max_candidates_num = max_candidates_num
        if success_rate is not None:
            self.success_log_rate_left = math.log(success_rate)
            self.num_merges_left = len(encoding_matrix.indptr)-1
        else:
            self.success_log_rate_left = None
            self.num_merges_left = None
        self.a = a

    def decode_belief_propagation(self, b, num_rounds):
        """
        Belief propagation algorithm to list-decode a.
        """
        cur_candidate_status_list = [True]
        f_init = F(self.base, self.p_err, b)
        f_list = [f_init]
        q_list = [Q(f_init, self.encoding_matrix)]
        r = R(self.encoding_matrix, self.encoded_a)
        stats = Stats(self.encoding_matrix, self.encoded_a, self.a, num_rounds, self.use_hints, self.use_forking)
        cur_round = -1

        while cur_round < num_rounds:
            num_candidates_start_of_round = len(cur_candidate_status_list)
            candidates_left = [i for i, status in enumerate(cur_candidate_status_list) if status]

            if not candidates_left:
                print("no candidates left")
                break

            for i in candidates_left:
                cur_round = cur_round + 1
                q = q_list[i]
                f = f_list[i]
                r.update(q)
                q.update(r)

                if self.not_going_anywhere(q) or self.not_going_anywhere(r):
                    cur_candidate_status_list[i] = False
                    q_list[i] = None
                    f_list[i] = None
                    stats.dead_end(i)
                    continue

                q_list[i] = q
                a_dist = r.calculate_new_distribution(f)
                stats.add_iteration(i, a_dist)

                if stats.should_get_hint(i, cur_round):
                    max_entropy_index = np.argmax(stats.entropies_list[0])
                    hint_value = self.a[max_entropy_index]
                    f_list = self.generate_forked_f(f, max_entropy_index, [hint_value])
                    q_list = q.fork(max_entropy_index, [hint_value], f_list)
                    stats.add_hint(max_entropy_index, hint_value)

                if stats.should_fork(i, cur_round):
                    max_entropy_index = np.argmax(stats.entropies_list[i])
                    forked_values = self.determine_forked_values(f, max_entropy_index)
                    forked_f_list = self.generate_forked_f(f, max_entropy_index, forked_values)
                    f_list = f_list + forked_f_list
                    q_list = q_list + q.fork(max_entropy_index, forked_values, forked_f_list)
                    num_forks = len(forked_values)
                    cur_candidate_status_list[i] = False
                    cur_candidate_status_list.extend(repeat(True, num_forks))
                    q_list[i] = None
                    f_list[i] = None
                    stats.add_fork(i, max_entropy_index, forked_values)

            if self.use_forking and len(candidates_left) > self.max_candidates_num:
                scores = [sum(stats.entropies_list[i]) if cur_candidate_status_list[i] else math.inf for i in range(num_candidates_start_of_round)]
                sorted_forks = sorted(range(len(scores)), key=lambda sub: scores[sub])
                best_forks = sorted_forks[:min(self.max_candidates_num, sum(i < math.inf for i in scores))]
                cur_candidate_status_list = [(i in best_forks) or (i >= num_candidates_start_of_round) for i in range(len(cur_candidate_status_list))]

        candidates_left = [i for i, status in enumerate(cur_candidate_status_list) if status]

        return candidates_left, stats

    def not_going_anywhere(self, vec):
        if np.isnan(vec.data).any():
            print("not going anywhere")
            return True
        return False

    def determine_forked_values(self, f, forked_index):
        return [s for s in range(self.base) if f[forked_index][s]]

    def decode_iteratively(self, b):
        f_init = F(self.base, self.p_err, b, use_log=True)
        index_to_id_map = dict()
        id_to_candidate_block_map = dict()

        for j in range(self.num_noise_symbols):
            i_list = self.encoding_matrix.j_to_i[j]
            base_candidates_mask = (f_init[j] != -math.inf)
            new_candidate_block = CandidateBlock(np.array([j]), np.array([[k] for k in range(self.base) if base_candidates_mask[k]]), f_init[j][base_candidates_mask])
            index_to_id_map[j] = j
            id_to_candidate_block_map[j] = new_candidate_block
            for i in i_list:
                j_list = self.encoding_matrix.indices[self.encoding_matrix.indptr[i]:self.encoding_matrix.indptr[i+1]]
                if max(j_list) == j:
                    ids = set([index_to_id_map[j_neighbor] for j_neighbor in j_list])
                    candidate_blocks_list = [id_to_candidate_block_map[id] for id in ids]
                    merged_candidate_block = self.merge_candidates(candidate_blocks_list, i, j_list)
                    id_to_candidate_block_map[merged_candidate_block.ID] = merged_candidate_block
                    for index in merged_candidate_block.indices:
                        if index != merged_candidate_block.ID:
                            index_to_id_map[index] = merged_candidate_block.ID
                            if index in ids:
                                del id_to_candidate_block_map[index]

                    iteration_success_log_prob = merged_candidate_block.prune(self.max_candidates_num, calc_max_iteration_log_prob(self.num_merges_left, self.success_log_rate_left))
                    if self.success_log_rate_left is not None:
                        self.success_log_rate_left -= iteration_success_log_prob
                        self.num_merges_left -= 1
            if (j % 50) == 0:
                print(j)
                print("max num candidates per block: " + str(max([len(candidate_block.candidates) for candidate_block in id_to_candidate_block_map.values()])))
                print("num candidate blocks: " + str(len(id_to_candidate_block_map)))

        final_candidate_block = self.merge_candidates(list(id_to_candidate_block_map.values()))
        print(final_candidate_block.candidate_log_probs)

        return final_candidate_block.sort_indices().candidates

    def merge_candidates(self, candidate_blocks_list, i=None, j_list=None):
        is_final_merge = (i is None) and (j_list is None)

        if len(candidate_blocks_list) == 1:
            new_indices = candidate_blocks_list[0].indices
            new_candidates_raw = candidate_blocks_list[0].candidates
            new_candidate_log_probs_raw = candidate_blocks_list[0].candidate_log_probs

        else:
            new_indices = np.array([index for candidate_block in candidate_blocks_list for index in candidate_block.indices])
            new_candidates_raw = np.array([[k for sub_array in new_candidate for k in sub_array] for new_candidate in itertools.product(*(candidate_block.candidates for candidate_block in candidate_blocks_list))])
            new_candidate_log_probs_raw = np.array([sum(new_candidate_log_probs) for new_candidate_log_probs in itertools.product(*(candidate_block.candidate_log_probs for candidate_block in candidate_blocks_list))])

        if not is_final_merge:
            j_to_index_map = [np.where(new_indices == j)[0][0] for j in j_list]
            indices_to_keep = [self.check_condition(c[j_to_index_map], i) for c in new_candidates_raw]
            if True not in indices_to_keep:
                raise "No candidates left"
        else:
            print("final merge")
            indices_to_keep = list(range(len(new_candidates_raw)))

        return CandidateBlock(new_indices, new_candidates_raw[indices_to_keep], new_candidate_log_probs_raw[indices_to_keep])

    def check_condition(self, candidate, i):
        return (np.inner(self.encoding_matrix[i], candidate) % self.base) == self.encoded_a[i]

class CandidateBlock(object):
    def __init__(self, indices, candidates, candidate_log_probs):
        self.ID = min(indices)
        self.indices = indices
        self.candidates = candidates
        self.candidate_log_probs = candidate_log_probs

    def prune(self, max_candidates_num, max_candidates_log_prob):
        assert (not (max_candidates_num and max_candidates_log_prob))
        if max_candidates_num is not None:
            if len(self.candidates) <= max_candidates_num:
                return None
            indices_to_keep = np.argpartition(self.candidate_log_probs, -max_candidates_num)[-max_candidates_num:]
        elif max_candidates_log_prob is not None:
            sorted_log_probs = np.sort(self.candidate_log_probs)
            total_log_prob = scipy.special.logsumexp(sorted_log_probs)
            num_indices_to_remove = 0
            cur_total_log_prob = total_log_prob
            while (cur_total_log_prob - total_log_prob > max_candidates_log_prob) and (cur_total_log_prob > sorted_log_probs[num_indices_to_remove]):
                next_total_log_prob = util.logminexp(cur_total_log_prob, sorted_log_probs[num_indices_to_remove])
                if next_total_log_prob - total_log_prob < max_candidates_log_prob:
                    break
                else:
                    cur_total_log_prob = next_total_log_prob
                    num_indices_to_remove += 1
            prune_rate = num_indices_to_remove/len(self.candidates)
            num_indices_to_keep = len(self.candidates) - num_indices_to_remove
            indices_to_keep = np.argpartition(self.candidate_log_probs, -num_indices_to_keep)[-num_indices_to_keep:]
        else:
            return None
        self.candidates = self.candidates[indices_to_keep]
        self.candidate_log_probs = self.candidate_log_probs[indices_to_keep]
        return (cur_total_log_prob - total_log_prob) if (max_candidates_log_prob is not None) else None

    def sort_indices(self):
        sorted_indices = np.sort(self.indices)
        arg_sorted_indices = np.argsort(self.indices)
        self.indices = sorted_indices
        self.candidates = self.candidates[:, arg_sorted_indices]
        return self

def calc_max_iteration_log_prob(num_merges_left, success_log_rate_left):
    if success_log_rate_left is None:
        return None
    use_product = True
    if use_product:
        return success_log_rate_left / num_merges_left
    else:
        pass
        return math.log(1 - (1 - math.exp(success_log_rate_left)) / num_merges_left)