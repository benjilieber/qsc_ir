import math

import numpy as np

import util
from result import Status, ListResultStatus, ResultType, Result


class Protocol(object):
    def __init__(self, cfg, a, b):
        self.cfg = cfg
        self.a = a
        self.b = b

    def run(self):
        pass

    def get_results(self,
                    key_length,  # in q-ary bits
                    a_guess_list,
                    a_key,
                    b_key_list,
                    leak_size,  # in bits
                    matrix_size,  # in bits
                    bob_communication_size,  # in bits
                    time):
        full_list_result = self.get_full_list_result(key_length=key_length,
                                                     a_guess_list=a_guess_list,
                                                     a_key=a_key,
                                                     b_key_list=b_key_list,
                                                     leak_size=leak_size,
                                                     matrix_size=matrix_size,
                                                     bob_communication_size=bob_communication_size,
                                                     time=time)
        checked_list_result = self.get_checked_list_result(key_length=key_length,
                                                           a_guess_list=a_guess_list,
                                                           a_key=a_key,
                                                           b_key_list=b_key_list,
                                                           leak_size=leak_size,
                                                           matrix_size=matrix_size,
                                                           bob_communication_size=bob_communication_size,
                                                           time=time)
        reduced_result = self.get_reduced_result(key_length=key_length,
                                                 a_guess_list=a_guess_list,
                                                 a_key=a_key,
                                                 b_key_list=b_key_list,
                                                 leak_size=leak_size,
                                                 matrix_size=matrix_size,
                                                 bob_communication_size=bob_communication_size,
                                                 time=time)
        return [full_list_result, checked_list_result, reduced_result]

    def get_full_list_result(self,
                             key_length,
                             a_guess_list,
                             a_key,
                             b_key_list,
                             leak_size,  # in bits
                             matrix_size,  # in bits
                             bob_communication_size,  # in bits
                             time):
        return self.get_result(result_type=ResultType.full_list,
                               key_length=key_length,
                               a_guess_list=a_guess_list,
                               a_key=a_key,
                               b_key_list=b_key_list,
                               leak_size=leak_size,
                               matrix_size=matrix_size,
                               bob_communication_size=bob_communication_size,
                               time=time)

    def get_checked_list_result(self,
                                key_length,  # in q-ary bits
                                a_guess_list,
                                a_key,
                                b_key_list,
                                leak_size,  # in bits
                                matrix_size,  # in bits
                                bob_communication_size,  # in bits
                                time):
        check_length = self.cfg.check_length

        if check_length != 0:
            check_matrix = np.random.choice(range(self.cfg.q), (key_length, self.cfg.check_length))
            check_value = np.matmul(a_key, check_matrix) % self.cfg.q
            guesses_check_values = [np.matmul(b_key, check_matrix) % self.cfg.q for b_key in b_key_list]
            guesses_passing_check = [i for i in range(len(b_key_list)) if
                                     util.hamming_single_block(guesses_check_values[i], check_value) == 0]
            b_key_list = [b_key_list[i] for i in guesses_passing_check]
            a_guess_list = [a_guess_list[i] for i in guesses_passing_check]
            leak_size_delta = self.cfg.check_length * math.log2(self.cfg.q)
            leak_size += leak_size_delta
            matrix_size_delta = key_length * self.cfg.check_length * math.log2(self.cfg.q)
            matrix_size += matrix_size_delta

        return self.get_result(result_type=ResultType.checked_list,
                               key_length=key_length,
                               a_guess_list=a_guess_list,
                               a_key=a_key,
                               b_key_list=b_key_list,
                               leak_size=leak_size,
                               matrix_size=matrix_size,
                               bob_communication_size=bob_communication_size,
                               time=time)

    def get_reduced_result(self,
                           key_length,  # in q-ary bits
                           a_guess_list,
                           a_key,
                           b_key_list,
                           leak_size,  # in bits
                           matrix_size,  # in bits
                           bob_communication_size,  # in bits
                           time):
        b_key_list = [list(b_key) for b_key in b_key_list]
        while len(a_guess_list) > 1:
            check_length = max(1, int(math.floor(math.log(len(a_guess_list), self.cfg.q))))
            check_matrix = np.random.choice(range(self.cfg.q), (key_length, check_length))
            check_value = np.matmul(a_key, check_matrix) % self.cfg.q
            guesses_check_values = [(np.matmul(b_key, check_matrix) % self.cfg.q).tolist() for b_key in b_key_list]
            guesses_passing_check = [i for i in range(len(b_key_list)) if
                                     util.hamming_single_block(list(guesses_check_values[i]), check_value) == 0]
            b_key_list = [b_key_list[i] for i in guesses_passing_check]
            a_guess_list = [a_guess_list[i] for i in guesses_passing_check]
            leak_size_delta = check_length * math.log2(self.cfg.q)
            leak_size += leak_size_delta
            matrix_size_delta = key_length * check_length * math.log2(self.cfg.q)
            matrix_size += matrix_size_delta
            bob_communication_size += math.log2(check_length)

        return self.get_result(result_type=ResultType.reduced,
                               key_length=key_length,
                               a_guess_list=a_guess_list,
                               a_key=a_key,
                               b_key_list=b_key_list,
                               leak_size=leak_size,
                               matrix_size=matrix_size,
                               bob_communication_size=bob_communication_size,
                               time=time)

    def get_result(self,
                   result_type,
                   key_length,  # in q-ary bits
                   a_guess_list,
                   a_key,
                   b_key_list,
                   leak_size,  # in bits
                   matrix_size,  # in bits
                   bob_communication_size,  # in bits
                   time):
        time_rate = time / self.cfg.N
        bob_communication_rate = bob_communication_size / self.cfg.N
        leak_rate = leak_size / self.cfg.N
        key_rate = math.log2(self.cfg.q) - leak_rate
        matrix_size_rate = matrix_size / self.cfg.N
        total_communication_rate = leak_rate + key_rate + matrix_size_rate

        if len(a_guess_list) == 0:
            return Result(cfg=self.cfg,
                          result_type=result_type,
                          result_status=Status.abort,
                          key_rate=key_rate,
                          leak_rate=leak_rate,
                          matrix_size_rate=matrix_size_rate,
                          bob_communication_rate=bob_communication_rate,
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
        ser_b_key = util.hamming_single_block(a_key, ml_b_key) / key_length
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
                      ser_b_key=ser_b_key,
                      ser_a_guess=ser_a_guess,
                      key_rate=key_rate,
                      leak_rate=leak_rate,
                      matrix_size_rate=matrix_size_rate,
                      bob_communication_rate=bob_communication_rate,
                      total_communication_rate=total_communication_rate,
                      time_rate=time_rate)
