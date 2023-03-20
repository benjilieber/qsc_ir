from enum import Enum

import numpy as np

import cfg
from ldpc import ldpc_cfg
from mb import mb_cfg
from polar import polar_cfg


class Status(Enum):
    success = 'success'
    fail = 'fail'
    abort = 'abort'

    def __str__(self):
        return self.value


class ResultType(Enum):
    full_list = 'full_list'
    checked_list = 'checked_list'
    reduced = 'full_reduce'

    def __str__(self):
        return self.value


class ListResultStatus(Enum):
    in_list_unique_max = 'in_list_unique_max'
    in_list_multi_max = 'in_list_multi_max'
    in_list_not_max = 'in_list_not_max'
    out_of_list_and_gt = 'out_of_list_and_gt'
    out_of_list_and_in_range = 'out_of_list_and_in_range'
    out_of_list_and_lt = 'out_of_list_and_lt'

    def __str__(self):
        return self.value


class Result(object):
    def __init__(self,
                 cfg,
                 result_type=None,
                 result_status=None,
                 list_result_status=None,
                 ser_b_key=None,
                 ser_a_guess=None,
                 key_rate=None,
                 leak_rate=None,
                 matrix_size_rate=None,
                 bob_communication_rate=None,
                 total_communication_rate=None,
                 time_rate=None,
                 result_list=None):
        self.cfg = cfg
        self.result_type = result_type
        if result_list is None:
            assert (result_status is not None)
            self.is_success = result_status == Status.success
            self.is_fail = result_status == Status.fail
            self.is_abort = result_status == Status.abort
            is_completed = not self.is_abort

            if result_status == Status.abort:
                self.in_list_unique_max = False
                self.in_list_multi_max = False
                self.in_list_not_max = False
                self.out_of_list_and_gt = False
                self.out_of_list_and_in_range = False
                self.out_of_list_and_lt = False
            else:
                assert (list_result_status is not None)
                self.in_list_unique_max = list_result_status == ListResultStatus.in_list_unique_max
                self.in_list_multi_max = list_result_status == ListResultStatus.in_list_multi_max
                self.in_list_not_max = list_result_status == ListResultStatus.in_list_not_max
                self.out_of_list_and_gt = list_result_status == ListResultStatus.out_of_list_and_gt
                self.out_of_list_and_in_range = list_result_status == ListResultStatus.out_of_list_and_in_range
                self.out_of_list_and_lt = list_result_status == ListResultStatus.out_of_list_and_lt

            self.ser_b_key_completed_only = ser_b_key if is_completed else None
            self.ser_b_key_fail_only = ser_b_key if self.is_fail else None
            self.ser_a_guess_completed_only = ser_a_guess if is_completed else None
            self.ser_a_guess_fail_only = ser_a_guess if self.is_fail else None
            self.key_rate = key_rate
            self.key_rate_completed_only = key_rate if is_completed else None
            self.key_rate_success_only = key_rate if self.is_success else None
            self.leak_rate = leak_rate
            self.leak_rate_completed_only = leak_rate if is_completed else None
            self.leak_rate_success_only = leak_rate if self.is_success else None
            self.matrix_size_rate = matrix_size_rate
            self.matrix_size_rate_success_only = matrix_size_rate if self.is_success else None
            self.bob_communication_rate = bob_communication_rate
            self.bob_communication_rate_success_only = bob_communication_rate if self.is_success else None
            self.total_communication_rate = total_communication_rate
            self.total_communication_rate_success_only = total_communication_rate if self.is_success else None
            self.time_rate = time_rate
            self.time_rate_success_only = time_rate if self.is_success else None
            self.sample_size = 1
        else:
            self.result_type = result_list[0].result_type
            assert ([cfg == result_list[i].cfg for i in range(len(result_list))])
            assert ([result_type == result_list[i].result_type for i in range(len(result_list))])

            self.is_success = np.mean([result.is_success for result in result_list])
            self.is_fail = np.mean([result.is_fail for result in result_list])
            self.is_abort = np.mean([result.is_abort for result in result_list])

            has_success = self.is_success > 0.0
            has_fail = self.is_fail > 0.0
            has_completed = has_success or has_fail

            success_result_list = [result for result in result_list if result.is_success]
            fail_result_list = [result for result in result_list if result.is_fail]
            completed_result_list = success_result_list + fail_result_list

            self.in_list_unique_max = np.mean([result.in_list_unique_max for result in completed_result_list])
            self.in_list_multi_max = np.mean([result.in_list_multi_max for result in completed_result_list])
            self.in_list_not_max = np.mean([result.in_list_not_max for result in completed_result_list])
            self.out_of_list_and_gt = np.mean([result.out_of_list_and_gt for result in completed_result_list])
            self.out_of_list_and_in_range = np.mean(
                [result.out_of_list_and_in_range for result in completed_result_list])
            self.out_of_list_and_lt = np.mean([result.out_of_list_and_lt for result in completed_result_list])

            self.ser_b_key_completed_only = np.mean(
                [result.ser_b_key_completed_only for result in completed_result_list]) if has_completed else None
            self.ser_b_key_fail_only = np.mean([result.ser_b_key_fail_only for result in fail_result_list]) if has_fail else None
            has_ser2 = has_completed and completed_result_list[0].ser_a_guess_completed_only is not None
            if has_ser2:
                self.ser_a_guess_completed_only = np.mean(
                    [result.ser_a_guess_completed_only for result in completed_result_list]) if has_completed else None
                self.ser_a_guess_fail_only = np.mean(
                    [result.ser_a_guess_fail_only for result in fail_result_list]) if has_fail else None
            else:
                self.ser_a_guess_completed_only = None
                self.ser_a_guess_fail_only = None

            self.key_rate = np.mean([result.key_rate for result in success_result_list] or [0.0])
            self.key_rate_completed_only = np.mean(
                [result.key_rate for result in completed_result_list]) if has_completed else None
            self.key_rate_success_only = np.mean(
                [result.key_rate for result in success_result_list]) if has_success else None

            self.leak_rate = np.mean([result.leak_rate for result in result_list])
            self.leak_rate_completed_only = np.mean(
                [result.leak_rate for result in completed_result_list]) if has_completed else None
            self.leak_rate_success_only = np.mean(
                [result.leak_rate for result in success_result_list]) if has_success else None

            self.matrix_size_rate = np.mean([result.matrix_size_rate for result in result_list])
            self.matrix_size_rate_success_only = np.mean(
                [result.matrix_size_rate for result in success_result_list]) if has_success else None

            self.bob_communication_rate = np.mean([result.bob_communication_rate for result in result_list])
            self.bob_communication_rate_success_only = np.mean(
                [result.bob_communication_rate for result in success_result_list]) if has_success else None

            self.total_communication_rate = np.mean([result.total_communication_rate for result in result_list])
            self.total_communication_rate_success_only = np.mean(
                [result.total_communication_rate for result in success_result_list]) if has_success else None

            self.time_rate = np.mean([result.time_rate for result in result_list])
            self.time_rate_success_only = np.mean(
                [result.time_rate for result in success_result_list]) if has_success else None

            self.sample_size = len(result_list)

    def get_cfg_dict(self):
        return self.cfg.log_dict()

    def get_output_dict(self):
        specific_dict = {"sample_size": self.sample_size,
                         "result_type": self.result_type,
                         "is_success": self.is_success,
                         "is_fail": self.is_fail,
                         "is_abort": self.is_abort,
                         "in_list_unique_max": self.in_list_unique_max,
                         "in_list_multi_max": self.in_list_multi_max,
                         "in_list_not_max": self.in_list_not_max,
                         "out_of_list_and_gt": self.out_of_list_and_gt,
                         "out_of_list_and_in_range": self.out_of_list_and_in_range,
                         "out_of_list_and_lt": self.out_of_list_and_lt,
                         "ser_b_key_completed_only": self.ser_b_key_completed_only,
                         "ser_b_key_fail_only": self.ser_b_key_fail_only,
                         "ser_a_guess_completed_only": self.ser_a_guess_completed_only,
                         "ser_a_guess_fail_only": self.ser_a_guess_fail_only,
                         "key_rate": self.key_rate,
                         "key_rate_completed_only": self.key_rate_completed_only,
                         "key_rate_success_only": self.key_rate_success_only,
                         "leak_rate": self.leak_rate,
                         "leak_rate_completed_only": self.leak_rate_completed_only,
                         "leak_rate_success_only": self.leak_rate_success_only,
                         "matrix_size_rate": self.matrix_size_rate,
                         "matrix_size_rate_success_only": self.matrix_size_rate_success_only,
                         "bob_communication_rate": self.bob_communication_rate,
                         "bob_communication_rate_success_only": self.bob_communication_rate_success_only,
                         "total_communication_rate": self.total_communication_rate,
                         "total_communication_rate_success_only": self.total_communication_rate_success_only,
                         "time_rate": self.time_rate,
                         "time_rate_success_only": self.time_rate_success_only}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return specific_dict

    def get_dict(self):
        return {**self.get_cfg_dict(), **self.get_output_dict()}

    def get_row(self):
        return list(self.get_dict().values())

    def get_header(self):
        return list(self.get_dict().keys())

    def get_cfg_row(self):
        return self.get_cfg_dict().values()

    def get_output_row(self):
        return self.get_output_dict().values()

    def get_cfg_header(self):
        return self.get_cfg_dict().keys()

    def get_output_header(self):
        return self.get_output_dict().keys()

    def __str__(self):
        cfg_string = "cfg:\n"
        for key, val in self.get_cfg_dict().items():
            cfg_string += "\t" + key + ": " + str(val) + "\n"
        if self.is_success is not None:
            output_string = "output:"
            for key, val in self.get_output_dict().items():
                output_string += "\t" + key + "=" + str(val) + "\n"
            return cfg_string.strip(", ") + output_string.strip("\n")
        else:
            return cfg_string.strip("\n")


def specific_log_header():
    return ["sample_size",
            "result_type",
            "is_success",
            "is_fail",
            "is_abort",
            "in_list_unique_max",
            "in_list_multi_max",
            "in_list_not_max",
            "out_of_list_and_gt",
            "out_of_list_and_in_range",
            "out_of_list_and_lt",
            "ser_b_key_completed_only",
            "ser_b_key_fail_only",
            "ser_a_guess_completed_only",
            "ser_a_guess_fail_only",
            "key_rate",
            "key_rate_completed_only",
            "key_rate_success_only",
            "leak_rate",
            "leak_rate_completed_only",
            "leak_rate_success_only",
            "matrix_size_rate",
            "matrix_size_rate_success_only",
            "bob_communication_rate",
            "bob_communication_rate_success_only",
            "total_communication_rate",
            "total_communication_rate_success_only",
            "time_rate",
            "time_rate_success_only"]


def get_header():
    return cfg.specific_log_header() + mb_cfg.specific_log_header() + ldpc_cfg.specific_log_header() + polar_cfg.specific_log_header() + specific_log_header()
