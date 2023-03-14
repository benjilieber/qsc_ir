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


class Result(object):
    def __init__(self, cfg, with_ml=None, result_status=None, is_success=None, is_fail=None, is_abort=None,
                 ser_completed_only=None, ser_fail_only=None, key_rate=None, key_rate_completed_only=None,
                 key_rate_success_only=None, leak_rate=None, leak_rate_completed_only=None, leak_rate_success_only=None,
                 matrix_size_rate=None, matrix_size_rate_success_only=None, bob_communication_rate=None,
                 bob_communication_rate_success_only=None, total_communication_rate=None,
                 total_communication_rate_success_only=None, time_rate=None, time_rate_success_only=None,
                 result_list=None, sample_size=None):
        self.cfg = cfg
        self.with_ml = with_ml
        if result_list is None:
            if result_status is not None:
                self.is_success = result_status == Status.success
                self.is_fail = result_status == Status.fail
                self.is_abort = result_status == Status.abort
            else:
                self.is_success = is_success
                self.is_fail = is_fail
                self.is_abort = is_abort

            self.ser_completed_only = ser_completed_only
            self.ser_fail_only = ser_fail_only
            self.key_rate = key_rate
            self.key_rate_completed_only = key_rate_completed_only
            self.key_rate_success_only = key_rate_success_only
            self.leak_rate = leak_rate
            self.leak_rate_completed_only = leak_rate_completed_only
            self.leak_rate_success_only = leak_rate_success_only
            self.matrix_size_rate = matrix_size_rate
            self.matrix_size_rate_success_only = matrix_size_rate_success_only
            self.bob_communication_rate = bob_communication_rate
            self.bob_communication_rate_success_only = bob_communication_rate_success_only
            self.total_communication_rate = total_communication_rate
            self.total_communication_rate_success_only = total_communication_rate_success_only
            self.time_rate = time_rate
            self.time_rate_success_only = time_rate_success_only or time_rate
            self.sample_size = sample_size or 1
        else:
            assert ([cfg == result_list[i].cfg for i in range(len(result_list))])
            assert ([with_ml == result_list[i].with_ml for i in range(len(result_list))])

            self.is_success = np.mean([result.is_success for result in result_list])
            self.is_fail = np.mean([result.is_fail for result in result_list])
            self.is_abort = np.mean([result.is_abort for result in result_list])

            has_success = self.is_success > 0.0
            has_fail = self.is_fail > 0.0
            has_completed = has_success or has_fail

            success_result_list = [result for result in result_list if result.is_success]
            fail_result_list = [result for result in result_list if result.is_fail]
            completed_result_list = success_result_list + fail_result_list

            self.ser_completed_only = np.mean(
                [result.ser_completed_only for result in completed_result_list]) if has_completed else None
            self.ser_fail_only = np.mean([result.ser_fail_only for result in fail_result_list]) if has_fail else None

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
                         "with_ml": self.with_ml,
                         "is_success": self.is_success,
                         "is_fail": self.is_fail,
                         "is_abort": self.is_abort,
                         "ser_completed_only": self.ser_completed_only,
                         "ser_fail_only": self.ser_fail_only,
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
        cfg_string = "cfg: "
        for key, val in self.get_cfg_dict().items():
            cfg_string += key + "=" + str(val) + ", "
        if self.is_success is not None:
            output_string = "output: "
            for key, val in self.get_output_dict().items():
                output_string += key + "=" + str(val) + ", "
            return cfg_string.strip(", ") + "\n" + output_string.strip(", ")
        else:
            return cfg_string.strip(", ")


def specific_log_header():
    return ["sample_size",
            "with_ml",
            "is_success",
            "is_fail",
            "is_abort",
            "ser_completed_only",
            "ser_fail_only",
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
