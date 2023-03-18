import math
import os
import time
from timeit import default_timer as timer

import numpy as np

import util
from polar.polar_encoder_decoder import PolarEncoderDecoder
from result import Status, ResultType, Result, ListResultStatus

from polar.scalar_distributions.qary_memoryless_distribution import QaryMemorylessDistribution

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
        start = timer()

        encoder_decoder = PolarEncoderDecoder(self.cfg)
        w, u = encoder_decoder.calculate_syndrome_and_complement(self.a)
        a_key = encoder_decoder.get_message_info_bits(u)
        x_b = self.b
        # x_b = np.mod(np.add(b, polarTransformOfQudits(self.cfg.q, w)), self.cfg.q)
        xy_vec_dist = self.xy_vec_dist_given_input_vec(x_b)
        frozen_vec = (encoder_decoder.get_message_frozen_bits(w) * (self.cfg.q - 1)) % self.cfg.q

        b_key_list, a_guess_list = encoder_decoder.list_decode(xy_vec_dist=xy_vec_dist,
                                                               frozen_values=frozen_vec)

        end = timer()

        full_list_result = self.get_result(time=end - start,
                                               result_type=ResultType.full_list,
                                               b_key_list=b_key_list,
                                               a_guess_list=a_guess_list,
                                               a_key=a_key)
        if self.cfg.check_length == 0:
            checked_list_result = self.get_result(time=end - start,
                                                  result_type=ResultType.checked_list,
                                                  b_key_list=b_key_list,
                                                  a_guess_list=a_guess_list,
                                                  a_key=a_key)
        else:
            check_matrix = np.random.choice(range(self.cfg.q), (self.cfg.num_info_indices, self.cfg.check_length))
            check_value = np.matmul(a_key, check_matrix) % self.cfg.q
            guesses_check_values = [np.matmul(b_key, check_matrix) % self.cfg.q for b_key in b_key_list]
            guesses_passing_check = [i for i in range(len(b_key_list)) if util.hamming_single_block(guesses_check_values[i], check_value) == 0]
            checked_list_result = self.get_result(time=end - start,
                                                  result_type=ResultType.checked_list,
                                                  b_key_list=[b_key_list[i] for i in guesses_passing_check],
                                                  a_guess_list=[a_guess_list[i] for i in guesses_passing_check],
                                                  a_key=a_key)

        return [full_list_result, checked_list_result]

    def xy_vec_dist_given_input_vec(self, received_vec):
        return self.cfg.xy_dist.makeQaryMemorylessVectorDistribution(self.cfg.N, received_vec, self.cfg.use_log)

    def x_vec_dist(self):
        x_dist = QaryMemorylessDistribution(self.cfg.q)
        x_dist.probs = [self.cfg.xy_dist.calcXMarginals()]
        x_vec_dist = x_dist.makeQaryMemorylessVectorDistribution(self.cfg.N, None, use_log=self.cfg.use_log)
        return x_vec_dist

    def get_result(self, time, result_type, b_key_list, a_guess_list, a_key):
        time_rate = time / self.cfg.N
        bob_communication_size = 1.0 / self.cfg.N
        if result_type == ResultType.full_list:
            key_rate = self.cfg.num_info_indices * math.log2(self.cfg.q) / self.cfg.N
            leak_rate = self.cfg.num_frozen_indices * math.log2(self.cfg.q) / self.cfg.N
            matrix_size_rate = 0.0
            total_communication_rate = self.cfg.num_frozen_indices / self.cfg.N
        else:
            key_rate = self.cfg.key_rate
            leak_rate = self.cfg.leak_rate
            matrix_size_rate = self.cfg.check_length * self.cfg.num_info_indices * math.log2(self.cfg.q) / self.cfg.N
            total_communication_rate = self.cfg.leak_rate + matrix_size_rate

        if len(b_key_list) == 0:
            return Result(cfg=self.cfg,
                          result_type=result_type,
                          result_status=Status.abort,
                          key_rate=key_rate,
                          leak_rate=leak_rate,
                          matrix_size_rate=matrix_size_rate,
                          bob_communication_rate=bob_communication_size,
                          total_communication_rate=total_communication_rate,
                          time_rate=time_rate)

        if self.cfg.p_err < 1 / self.cfg.q:
            actual_pm = util.closeness_single_block(self.b, self.a)
            pm_list = [util.closeness_single_block(self.b, a_guess) for a_guess in a_guess_list]
        else:
            actual_pm = util.hamming_single_block(self.b, self.a)
            pm_list = [util.hamming_single_block(self.b, a_guess) for a_guess in a_guess_list]
        se_list = [util.hamming_single_block(a_key, b_key) for b_key in b_key_list]
        ml_index = np.argmax(pm_list)
        ml_b_key = b_key_list[ml_index]
        ser_b_key = util.hamming_single_block(a_key, ml_b_key) / self.cfg.num_info_indices
        ml_a_guess = a_guess_list[ml_index]
        ser_a_guess = util.hamming_single_block(self.a, ml_a_guess) / self.cfg.N

        if ser_b_key == 0.0:
            result_status = Status.success
        else:
            result_status = Status.fail
        if 0 in se_list:
            if actual_pm == np.max(pm_list):
                if pm_list.count(actual_pm) == 1:
                    list_result_status = ListResultStatus.in_list_unique_max
                else:
                    list_result_status = ListResultStatus.in_list_multi_max
            else:
                list_result_status = ListResultStatus.in_list_not_max
        else:
            if actual_pm > np.max(pm_list):
                list_result_status = ListResultStatus.out_of_list_and_gt
            elif actual_pm >= np.min(pm_list):
                list_result_status = ListResultStatus.out_of_list_and_in_range
            else:
                list_result_status = ListResultStatus.out_of_list_and_lt

        return Result(cfg=self.cfg,
                      result_type=result_type,
                      result_status=result_status,
                      list_result_status=list_result_status,
                      ser=ser_b_key,
                      ser2=ser_a_guess,
                      key_rate=key_rate,
                      leak_rate=leak_rate,
                      matrix_size_rate=matrix_size_rate,
                      bob_communication_rate=bob_communication_size,
                      total_communication_rate=total_communication_rate,
                      time_rate=time_rate)
