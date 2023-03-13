import os
import time
from random import random
from timeit import default_timer as timer
import numpy as np

import util
from result import Result, Status


class PolarProtocol(object):
    def __init__(self, cfg, a, b):
        self.cfg = cfg
        self.a = a
        self.b = b

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.total_leak = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1

    def run(self):
        make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(self.xy_dist, use_log)

        start = timer()

        # def irSimulation(q, length, simulateChannel, make_xyVectorDistribution, numberOfTrials,
        #                  frozenSet, maxListSize=1, checkSize=0,
        #                  use_log=False, verbosity=0):

        w, u = self.calculate_syndrome_and_complement(a)
        a_key = self.get_message_info_bits(u)
        x_b = b
        # x_b = np.mod(np.add(b, polarTransformOfQudits(self.q, w)), self.q)
        frozen_values = (self.get_message_frozen_bits(w) * (self.q - 1)) % self.q
        # print(w)
        # print(self.frozenSet)
        # print(frozen_values)

        # if list_size == 1:
        #     b_key = self.decode(xVectorDistribution, make_xyVectorDistribution(x_b))
        # else:
        check_matrix = np.random.choice(range(self.q), (self.k, check_size))
        check_value = np.matmul(a_key, check_matrix) % self.q
        b_key, prob_result = self.listDecode(make_xyVectorDistribution(x_b), frozenValues=frozen_values,
                                             maxListSize=list_size, check_matrix=check_matrix, check_value=check_value,
                                             actualInformation=a_key, verbosity=verbosity)

        return a_key, b_key, prob_result

        probResultList.append(prob_result)

        if not np.array_equal(a_key, b_key):
            badKeys += 1
            badSymbols += util.hamming_single_block(a_key, b_key)

        assert (len(a_key) == length - len(frozenSet))
        # rate = (math.log(q, 2)*len(a_key) - math.log(maxListSize, 2))/length
        rate = (math.log2(q) * len(a_key) - math.log2(maxListSize)) / length
        frame_error_prob = badKeys / numberOfTrials
        symbol_error_prob = badSymbols / (numberOfTrials * encDec.length)

        if verbosity:
            print("Rate: ", rate)
            print("Frame error probability = ", badKeys, "/", numberOfTrials, " = ", frame_error_prob)
            print("Symbol error probability = ", badSymbols, "/ (", numberOfTrials, " * ", encDec.length, ") = ",
                  symbol_error_prob)

        end = timer()
        time_rate = (end - start) / self.N

        ml_result = self.get_result(time=end - start, is_ml=True)

        if self.cfg.verbosity:
            print("hamming distance x and most probable x', pre-reducing: " + str(
                util.hamming_multi_block(self.bob.a_candidates[np.argmin(self.bob.a_candidates_errors)], self.alice.a)))

        while self.cur_candidates_num > 1:
            self.run_single_round(encode_new_block=False)

        if self.cfg.verbosity:
            print("actual_closeness: ", util.closeness_single_block(self.a, self.b))
        non_ml_result = self.get_result(time=end - start, is_ml=False)

        return [non_ml_result, ml_result]

    def make_xy_vec_dist(self, received_vec):
        return self.xy_dist.makeQaryMemorylessVectorDistribution(self.N, received_vec, self.cfg.use_log)