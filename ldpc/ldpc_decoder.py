import copy
import numpy as np
import math
from itertools import repeat

from ldpc.f import F
from q import Q
from r import R
from stats import Stats


class LdpcDecoder(object):

    def __init__(self, base, p_err, encoding_matrix, encoded_a, a=None, use_forking=False, use_hints=False, max_candidates_num=1):
        self.base = base
        self.p_err = p_err
        self.encoded_a = encoded_a
        assert(not (use_forking and use_hints))
        self.use_forking = use_forking
        self.use_hints = use_hints
        self.encoding_matrix = encoding_matrix
        if use_forking:
            self.max_candidates_num = max_candidates_num
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
