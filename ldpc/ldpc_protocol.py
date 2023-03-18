import math
import os
import time
import numpy as np

import util
from ldpc.ldpc_cfg import Decoder
from ldpc.ldpc_decoder import LdpcDecoder
from ldpc.ldpc_generator import LdpcGenerator
from result import ResultType, Result, Status, ListResultStatus


class LdpcProtocol(object):
    def __init__(self, cfg, a, b):
        self.cfg = cfg
        self.a = a
        self.b = b

        np.random.seed([os.getpid(), int(str(time.time() % 1)[2:10])])

        self.total_leak = 0
        self.total_communication_size = 0

        self.cur_candidates_num = 1

    def run(self):
        assert(self.cfg.L == 1)

        start = time.time()
        code_gen = LdpcGenerator(self.cfg)
        encoding_matrix = code_gen.generate_gallagher_matrix(self.cfg.syndrome_length)
        encoded_a = encoding_matrix * self.a
        decoder = LdpcDecoder(self.cfg, encoding_matrix, encoded_a)
        if self.cfg.decoder == Decoder.bp:
            _, stats = decoder.decode_belief_propagation(self.b, 20)
        else:
            raise "No support for iterative ldpc decoder"
        end = time.time()
        return self.get_result(end-start, a_guess_list=stats.a_guess_list)

    def get_result(self, time, a_guess_list):
        assert (len(a_guess_list) == 1)
        a_guess = a_guess_list[0]

        time_rate = time / self.cfg.N
        bob_communication_size = 1.0 / self.cfg.N
        leak_rate = self.cfg.syndrome_length * math.log2(self.cfg.q) / self.cfg.N
        key_rate = math.log2(self.cfg.q) - leak_rate
        matrix_size_rate = self.cfg.syndrome_length * self.cfg.sparsity * math.log2(self.cfg.N) * math.log2(self.cfg.q - 1)
        total_communication_rate = matrix_size_rate + self.cfg.syndrome_length * math.log2(self.cfg.q)

        if a_guess is None:
            return Result(cfg=self.cfg,
                          result_type=ResultType.full_list,
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
        se_list = [util.hamming_single_block(self.a, a_guess) for a_guess in a_guess_list]
        ml_index = np.argmax(pm_list)
        ml_a_guess = a_guess_list[ml_index]
        ser_a_guess = util.hamming_single_block(self.a, ml_a_guess) / self.cfg.N

        if ser_a_guess == 0.0:
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
                      result_type=ResultType.full_list,
                      result_status=result_status,
                      list_result_status=list_result_status,
                      ser=ser_a_guess,
                      key_rate=key_rate,
                      leak_rate=leak_rate,
                      matrix_size_rate=matrix_size_rate,
                      bob_communication_rate=bob_communication_size,
                      total_communication_rate=total_communication_rate,
                      time_rate=time_rate)