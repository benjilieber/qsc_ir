import math

import numpy as np
from scipy.stats import entropy

import util


class Stats(object):
    def __init__(self, encoding_matrix, encoded_a, a, num_rounds, use_hints, use_forking):
        self.encoding_matrix = encoding_matrix
        self.encoded_a = encoded_a
        self.a = a
        self.num_rounds = num_rounds
        self.use_hints = use_hints
        self.use_forking = use_forking
        self.a_guess_list = [None]
        self.encoding_error_trajectory_list = [[]]
        if a is not None:
            self.trajectory_list = [[]]
        self.entropies_list = [None]
        self.entropy_sum_trajectory_list = [[]]

    def dead_end(self, i):
        self.a_guess_list[i] = None
        self.encoding_error_trajectory_list[i].append("dead end")
        if self.a is not None:
            self.trajectory_list[i].append("dead end")
        self.entropies_list[i] = None
        self.entropy_sum_trajectory_list[i].append("dead end")

    def add_iteration(self, i, a_dist):
        a_guess = np.argmax(a_dist, axis=1)
        self.a_guess_list[i] = a_guess
        self.encoding_error_trajectory_list[i].append(
            util.hamming_single_block(self.encoding_matrix * a_guess, self.encoded_a))
        if self.a is not None:
            self.trajectory_list[i].append(util.hamming_single_block(self.a, a_guess))
        entropies = [entropy(a_dist_i, base=2) for a_dist_i in a_dist]
        self.entropies_list[i] = entropies
        a_guess_entropy_sum = sum(entropies)
        self.entropy_sum_trajectory_list[i].append(a_guess_entropy_sum)

    def add_hint(self, hint_index, hint_value):
        hint_name = "hint " + str(hint_index) + " - " + str(hint_value)
        self.encoding_error_trajectory_list[0].append(hint_name)
        if self.a is not None:
            self.trajectory_list[0].append(hint_name)
        self.entropy_sum_trajectory_list[0].append(hint_name)

    def add_fork(self, i, forked_index, forked_values):
        for forked_value in forked_values:
            fork_name = "fork " + str(forked_index) + " - " + str(forked_value)
            self.a_guess_list.append(self.a_guess_list[i])
            self.encoding_error_trajectory_list.append(self.encoding_error_trajectory_list[i] + [fork_name])
            if self.a is not None:
                self.trajectory_list.append(self.trajectory_list[i] + [fork_name])
            self.entropies_list.append(self.entropies_list[i])
            self.entropy_sum_trajectory_list.append(self.entropy_sum_trajectory_list[i] + [fork_name])

        fork_name = "fork " + str(forked_index)
        self.a_guess_list[i] = None
        self.encoding_error_trajectory_list[i].append(fork_name)
        if self.a is not None:
            self.trajectory_list[i].append(fork_name)
        self.entropies_list[i] = None
        self.entropy_sum_trajectory_list[i].append(fork_name)

    def should_get_hint(self, i, cur_round):
        if self.use_hints and self.num_rounds - cur_round > self.base and cur_round > 10:
            last_10_index = [index for index, entropy_sum in enumerate(self.entropy_sum_trajectory_list[i]) if
                             not isinstance(entropy_sum, str)][-11]
            number_rounds_since_last_hint = len(self.entropy_sum_trajectory_list[i]) - max(
                [index for index, entropy_sum in enumerate(self.entropy_sum_trajectory_list[i]) if
                 isinstance(entropy_sum, str)] or [-math.inf])
            if number_rounds_since_last_hint >= 5 and self.entropy_sum_trajectory_list[i][-1] > 0.5 and abs(
                    self.entropy_sum_trajectory_list[i][-1] - self.entropy_sum_trajectory_list[i][last_10_index]) <= 10:
                return True
        return False

    def should_fork(self, i, cur_round):
        if self.use_forking and self.num_rounds - cur_round > self.base and cur_round > 10:
            last_10_index = [index for index, entropy_sum in enumerate(self.entropy_sum_trajectory_list[i]) if
                             not isinstance(entropy_sum, str)][-11]
            number_rounds_since_last_fork = len(self.entropy_sum_trajectory_list[i]) - max(
                [index for index, entropy_sum in enumerate(self.entropy_sum_trajectory_list[i]) if
                 isinstance(entropy_sum, str)] or [-math.inf])
            if number_rounds_since_last_fork >= 5 and self.entropy_sum_trajectory_list[i][-1] > 0.5 and abs(
                    self.entropy_sum_trajectory_list[i][-1] - self.entropy_sum_trajectory_list[i][last_10_index]) <= 10:
                return True
        return False
