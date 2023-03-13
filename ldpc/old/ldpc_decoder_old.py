import copy

import numpy as np
import math
from scipy.stats import entropy
from itertools import repeat

import util
from mb.encoder import Encoder


class LdpcDecoder(object):

    def __init__(self, m, p_err, encoding_matrix, encoded_a, use_forking=False, use_hints=False, max_candidates_num=1):
        self.m = m  # basis field size
        self.p_err = p_err
        self.encoded_a = encoded_a
        assert(not (use_forking and use_hints))
        self.use_forking = use_forking
        self.use_hints = use_hints
        self.encoding_matrix = encoding_matrix
        self.num_checks = encoding_matrix.shape[0]  # "= m, i"
        self.num_noise_symbols = encoding_matrix.shape[1]  # "= n, j"

        self.i_to_j = [encoding_matrix.indices[cur:next] for cur, next in zip(encoding_matrix.indptr, encoding_matrix.indptr[1:])]
        self.j_to_i = [[] for _ in range(self.num_noise_symbols)]
        for i, p in enumerate(self.i_to_j):
            for j in p:
                self.j_to_i[j] += [i]
        self.sparsity = len(max(map(len, self.i_to_j)))
        self.t_sparsity = len(max(map(len, self.j_to_i)))

        self.encoder = Encoder(m, self.num_noise_symbols)
        if use_forking:
            self.max_candidates_num = max_candidates_num

    def decode_belief_propagation(self, b, num_rounds, a=None):
        """
        Belief propagation algorithm to list-decode a.
        """
        # init
        cur_candidate_status_list = [True]
        f_init = self.calculate_f(b)  # (num_noise_symbols, m)
        f_list = [f_init]
        q_list = [np.array([[f_init[j] if j != -1 else [-1] * self.m for j in self.i_to_j[i]] for i in range(self.num_checks)])]
        # TODO: optimize this - see https://scicomp.stackexchange.com/questions/35242/fast-nonzero-indices-per-row-column-for-sparse-2d-numpy-array
        trajectory_list = [[]]
        encoding_error_trajectory_list = [[]]
        entropy_sum_trajectory_list = [[]]
        a_guess_list = [None]
        entropies_list = [None]
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
                r = self.calculate_new_r(q)  # (num_checks, sparsity, m)
                q = self.calculate_new_q(f, r)  # (num_checks, sparsity, m)

                if self.not_going_anywhere(q) or self.not_going_anywhere(r):
                    cur_candidate_status_list[i] = False
                    q_list[i] = None
                    f_list[i] = None
                    encoding_error_trajectory_list[i].append("dead end")
                    entropy_sum_trajectory_list[i].append("dead end")
                    if a is not None:
                        trajectory_list[i].append("dead end")
                    a_guess_list[i] = None
                    entropies_list[i] = None
                    continue

                q_list[i] = q
                a_dist = self.calculate_new_distribution(f, r)
                a_guess = np.argmax(a_dist, axis=1)
                a_guess_list[i] = a_guess
                encoding_error_trajectory_list[i].append(
                    util.hamming_single_block(self.encoding_matrix * a_guess, self.encoded_a))
                if a is not None:
                    trajectory_list[i].append(util.hamming_single_block(a, a_guess))
                entropies = [entropy(a_dist_i, base=2) for a_dist_i in a_dist]
                entropies_list[i] = entropies
                a_guess_entropy_sum = sum(entropies)
                entropy_sum_trajectory_list[i].append(a_guess_entropy_sum)

                if self.use_hints and num_rounds - cur_round > self.m and cur_round > 10:
                    last_10_index = [index for index, entropy_sum in enumerate(entropy_sum_trajectory_list[i]) if
                                     not isinstance(entropy_sum, str)][-11]
                    number_rounds_since_last_hint = len(entropy_sum_trajectory_list[i]) - max(
                        [index for index, entropy_sum in enumerate(entropy_sum_trajectory_list[i]) if
                         isinstance(entropy_sum, str)] or [-math.inf])
                    if number_rounds_since_last_hint >= 5 and entropy_sum_trajectory_list[i][-1] > 0.5 and abs(
                        entropy_sum_trajectory_list[i][-1] - entropy_sum_trajectory_list[i][last_10_index]) <= 10:
                        max_entropy_index = np.argmax(entropies)
                        hint_value = a[max_entropy_index]
                        q_list = self.generate_forked_q(q, max_entropy_index, [hint_value])
                        f_list = self.generate_forked_f(f, max_entropy_index, [hint_value])
                        hint_name = "hint " + str(max_entropy_index) + " - " + str(hint_value)
                        encoding_error_trajectory_list[0].append(hint_name)
                        if a is not None:
                            trajectory_list[0].append(hint_name)
                        entropy_sum_trajectory_list[0].append(hint_name)


                if self.use_forking and num_rounds - cur_round > self.m and cur_round > 10:
                    last_10_index = [index for index, entropy_sum in enumerate(entropy_sum_trajectory_list[i]) if not isinstance(entropy_sum, str)][-11]
                    number_rounds_since_last_fork = len(entropy_sum_trajectory_list[i]) - max([index for index, entropy_sum in enumerate(entropy_sum_trajectory_list[i]) if isinstance(entropy_sum, str)] or [-math.inf])
                    if number_rounds_since_last_fork >= 5 and entropy_sum_trajectory_list[i][-1] > 0.5 and abs(entropy_sum_trajectory_list[i][-1] - entropy_sum_trajectory_list[i][last_10_index]) <= 10:
                        max_entropy_index = np.argmax(entropies)
                        forked_values = self.determine_forked_values(f, max_entropy_index)
                        f_list = f_list + self.generate_forked_f(f, max_entropy_index, forked_values)
                        q_list = q_list + self.generate_forked_q(q, max_entropy_index, forked_values)

                        num_forks = len(forked_values)
                        for forked_value in forked_values:
                            fork_name = "fork " + str(max_entropy_index) + " - " + str(forked_value)
                            encoding_error_trajectory_list.append(encoding_error_trajectory_list[i] + [fork_name])
                            if a is not None:
                                trajectory_list.append(trajectory_list[i] + [fork_name])
                            entropy_sum_trajectory_list.append(entropy_sum_trajectory_list[i] + [fork_name])
                            entropies_list.append(entropies_list[i])
                            a_guess_list.append(a_guess_list[i])

                        cur_candidate_status_list[i] = False
                        cur_candidate_status_list.extend(repeat(True, num_forks))
                        encoding_error_trajectory_list[i].append("forking " + str(str(max_entropy_index)))
                        entropy_sum_trajectory_list[i].append("forking " + str(str(max_entropy_index)))
                        entropies_list[i] = None
                        if a is not None:
                            trajectory_list[i].append("forking " + str(str(max_entropy_index)))
                        q_list[i] = None
                        f_list[i] = None
                        a_guess_list[i] = None

            if self.use_forking and len(candidates_left) > self.max_candidates_num:
                # Pick the best max_candidates_num forks
                # forked_a_guess_entropy_sum_list = np.array(forked_a_guess_entropy_sum_list)
                # if np.isnan(forked_a_guess_entropy_sum_list).all():
                #     best_fork = 0
                # else:
                #     best_fork = np.nanargmin(forked_a_guess_entropy_sum_list)
                # heuristic 1: encoding error
                # scores = [util.hamming_single_block(self.encoding_matrix.encode(a_guess_list[i]), self.encoded_a) if cur_candidate_status_list[i] else math.inf for i in range(num_candidates_start_of_round)]
                # heuristic 2: entropy sum
                scores = [sum(entropies_list[i]) if cur_candidate_status_list[i] else math.inf for i in range(num_candidates_start_of_round)]
                sorted_forks = sorted(range(len(scores)), key=lambda sub: scores[sub])
                best_forks = sorted_forks[:min(self.max_candidates_num, sum(i < math.inf for i in scores))]
                cur_candidate_status_list = [(i in best_forks) or (i >= num_candidates_start_of_round) for i in range(len(cur_candidate_status_list))]

            # if any([f[j][a[j]] == 0.0 for j in range(self.num_noise_symbols)]):
            #     print("Wrong values were decided")

            # f = self.calculate_new_f(f, q)

        candidates_left = [i for i, status in enumerate(cur_candidate_status_list) if status]
        # for i in candidates_left:
        #     graph = Graph()
        #     graph.addEdges(self.i_to_j, self.num_noise_symbols)
        #     graph.removeNodes(entropies_list[i], 0.2)
        #     graph.visualize()

        return candidates_left, a_guess_list, encoding_error_trajectory_list, trajectory_list, entropy_sum_trajectory_list, entropies, a_guess_entropy_sum

    def calculate_f(self, b):
        return np.array([[self.calculate_f_helper(s, b_i) for s in range(self.m)] for b_i in b])

    def calculate_f_helper(self, s, b_i):
        f_raw = self.p_err if b_i == s else (1 - self.p_err) / (self.m - 1)
        return f_raw

    def calculate_new_r(self, q):
        """
        Check probabilities: r[i][j][s] is the probability that check i is satisfied (a * encoding_matrix[:][i] == encoded_a[i])
        if a_j == s and the other noise symbols have a separable distribution given by the probabilities q[i][J[i]\{j}].
        """
        sigma = np.full((self.num_checks, self.sparsity, self.m), -1.0)
        # Fill J[i] values for all i
        for i, j_list in enumerate(self.i_to_j):
            sigma[i] = self.calculate_sigma_helper(q[i], self.encoding_matrix[i], j_list)

        rho = np.full((self.num_checks, self.sparsity, self.m), -1.0)
        # Fill J[i] values for all i
        for i, j_list in enumerate(self.i_to_j):
            rho[i] = self.calculate_rho_helper(q[i], self.encoding_matrix[i], j_list)

        r = [self.calculate_r_helper(sigma[i], rho[i], self.encoded_a[i], self.encoding_matrix[i], self.i_to_j[i]) for i in
             range(self.num_checks)]

        return r

    def calculate_sigma_helper(self, q_row, encoding_matrix_row, j_list):
        sigma_row = np.full((1, self.sparsity, self.m), -1.0)

        for j_ord, j in enumerate(j_list):
            if j == -1:
                continue

            enc_coef = encoding_matrix_row[j]

            if j_ord == 0:
                sigma_row[0] = [
                    np.sum([q_row[0][t] for t in range(self.m) if ((enc_coef * t) % self.m == s)] or [0]) for s
                    in
                    range(self.m)]

            else:
                sigma_row[j_ord] = [np.sum(
                    [sigma_row[j_ord - 1, (s - enc_coef * t) % self.m] * q_row[j_ord][t] for t in range(self.m)]) for s in
                    range(self.m)]

        return sigma_row

    def calculate_rho_helper(self, q_row, encoding_matrix_row, j_list):
        rho_row = np.full((1, self.sparsity, self.m), -1.0)

        started = False
        for j_ord, j in zip(range(self.sparsity - 1, -1, -1), reversed(j_list)):
            if j == -1:
                continue

            enc_coef = encoding_matrix_row[j]

            if not started:
                started = True
                rho_row[-1] = [
                    np.sum([q_row[-1][t] for t in range(self.m) if ((enc_coef * t) % self.m == s)] or [0]) for s
                    in range(self.m)]

            else:
                rho_row[j_ord] = [np.sum(
                    [rho_row[j_ord + 1, (s - enc_coef * t) % self.m] * q_row[j_ord][t] for t in range(self.m)]) for s in
                    range(self.m)]

        return rho_row

    def calculate_r_helper(self, sigma_row, rho_row, encoded_value, encoding_row, j_list):
        return [[np.sum([(sigma_row[j_ord - 1,
                              (encoded_value - encoding_row[j] * s - t) % self.m] if j_ord - 1 >= 0 else (
                ((encoded_value - encoding_row[j] * s - t) % self.m) == 0))
                       * (rho_row[j_ord + 1, t] if (j_ord + 1 < self.sparsity and j_list[j_ord + 1] != -1) else (t == 0))
                       for t in range(self.m)]) for s in range(self.m)] if j != -1 else [-1.0] * self.m for j_ord, j in enumerate(j_list)]

    def calculate_new_q(self, f, r):
        """
        Symbol probabilities: q[i][j][s] is the probability that a_j == s given the information obtained via
        checks I[j]\{i}.
        """
        q = np.array(
            [[[f[j][s] * np.prod([r[k][np.where(self.i_to_j[k] == j)[0][0]][s] for k in self.j_to_i[j] if k != i and k != -1]) for s in range(self.m)]
              if j != -1 else [-1.0] * self.m for
              j in
              self.i_to_j[i]] for i in range(self.num_checks)])
        q_sums = q.sum(axis=2)
        q_sums = (q_sums < 0.0).astype(float) + (q_sums >= 0.0).astype(float) * q_sums
        q_normalized = q / q_sums[:, :, np.newaxis]
        return q_normalized

    def calculate_new_distribution(self, f, r):
        return [[f[j][s] * np.prod([r[i][np.where(self.i_to_j[i] == j)[0][0]][s] for i in self.j_to_i[j] if i != -1]) for s in range(self.m)] for j in
                range(self.num_noise_symbols)]

    def not_going_anywhere(self, vec):
        if np.isnan(vec).any() or (np.sum(vec, axis=2) == 0.0).any():
            print("not going anywhere")
            return True
        return False

    def determine_forked_values(self, f, forked_index):
        return [s for s in range(self.m) if f[forked_index][s]]

    def generate_forked_f(self, f, forked_index, forked_values):
        forked_f_list = []
        for s in forked_values:
            forked_f = copy.deepcopy(f)
            forked_index_dist = [0.0 + 1.0 * (s == s_i) for s_i in range(self.m)]
            forked_f[forked_index] = forked_index_dist
            forked_f_list.append(forked_f)
        return forked_f_list

    def generate_forked_q(self, q, forked_index, forked_values):
        forked_q_list = []
        for s in forked_values:
            forked_q = copy.deepcopy(q)
            forked_index_dist = [0.0 + 1.0 * (s == s_i) for s_i in range(self.m)]
            for i in self.j_to_i[forked_index]:
                if i == -1:
                    break
                forked_q[i][np.where(self.i_to_j[i] == forked_index)[0][0]] = forked_index_dist
            forked_q_list.append(forked_q)
        return forked_q_list

    def calculate_new_f(self, f, q):
        """
        My addition to the algorithm. If we can eliminate values of indices, do it.
        :param f:
        :param q:
        :return:
        """
        q_max = [[max([q[i][np.where(self.i_to_j[i] == j)[0][0]][s] for i in self.j_to_i[j]]) for s in range(self.m)] for j in range(self.num_noise_symbols)]
        # new_f_raw = np.array(
        #     [[0.0 if q_max[j][s] == 0.0 else f[j][s] for s in range(self.m)] for j in range(self.num_noise_symbols)])
        new_f_raw = np.array(
            [[0.0 if q_max[j][s] < 0.0001 else f[j][s] for s in range(self.m)] for j in range(self.num_noise_symbols)])
        new_f_sums = new_f_raw.sum(axis=1)
        new_f = new_f_raw / new_f_sums[:, np.newaxis]
        if np.count_nonzero(f == 1.0) != np.count_nonzero(new_f == 1.0):
            print("f changed from " + str(np.count_nonzero(f == 1.0)) + " to " + str(np.count_nonzero(new_f == 1.0)))
        return new_f
