import numpy as np
import pandas as pd
import glob

from result import Result, Status
from cfg import CodeStrategy
from mb.mb_cfg import MbCfg, PruningStrategy, RoundingStrategy

class MbResult(Result):
    def __init__(self, cfg, with_ml=None, result_status=None, is_success=None, is_fail=None, is_abort=None, ser_completed_only=None, ser_fail_only=None, key_rate=None, key_rate_completed_only=None, key_rate_success_only=None, leak_rate=None, leak_rate_completed_only=None, leak_rate_success_only=None, matrix_size_rate=None, matrix_size_rate_success_only=None, bob_communication_rate=None,
                 bob_communication_rate_success_only=None, total_communication_rate=None, total_communication_rate_success_only=None, time_rate=None, time_rate_success_only=None, result_list=None, sample_size=None):
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

            self.ser_completed_only = np.mean([result.ser_completed_only for result in completed_result_list]) if has_completed else None
            self.ser_fail_only = np.mean([result.ser_fail_only for result in fail_result_list]) if has_fail else None

            self.key_rate = np.mean([result.key_rate for result in success_result_list] or [0.0])
            self.key_rate_completed_only = np.mean([result.key_rate for result in completed_result_list]) if has_completed else None
            self.key_rate_success_only = np.mean([result.key_rate for result in success_result_list]) if has_success else None

            self.leak_rate = np.mean([result.leak_rate for result in result_list])
            self.leak_rate_completed_only = np.mean([result.leak_rate for result in completed_result_list]) if has_completed else None
            self.leak_rate_success_only = np.mean([result.leak_rate for result in success_result_list]) if has_success else None

            self.matrix_size_rate = np.mean([result.matrix_size_rate for result in result_list])
            self.matrix_size_rate_success_only = np.mean([result.matrix_size_rate for result in success_result_list]) if has_success else None

            self.bob_communication_rate = np.mean([result.bob_communication_rate for result in result_list])
            self.bob_communication_rate_success_only = np.mean([result.bob_communication_rate for result in success_result_list]) if has_success else None

            self.total_communication_rate = np.mean([result.total_communication_rate for result in result_list])
            self.total_communication_rate_success_only = np.mean([result.total_communication_rate for result in success_result_list]) if has_success else None

            self.time_rate = np.mean([result.time_rate for result in result_list])
            self.time_rate_success_only = np.mean([result.time_rate for result in success_result_list]) if has_success else None

            self.sample_size = len(result_list)

    def get_cfg_row(self):
        return [self.cfg.q, self.cfg.N, self.cfg.block_length, self.cfg.num_blocks, self.cfg.p_err, self.cfg.success_rate,
                self.cfg.goal_candidates_num, self.cfg.max_candidates_num,
                self.cfg.max_num_indices_to_encode, self.cfg.fixed_number_of_encodings, str(self.cfg.code_strategy), str(self.cfg.rounding_strategy), str(self.cfg.pruning_strategy), self.cfg.radius_picking, self.cfg.encoding_sample_size, self.cfg.theoretic_key_rate]

    def __str__(self):
        cfg_string = "cfg: "
        for key, val in zip(get_cfg_header(), self.get_cfg_row()):
            cfg_string += key + "=" + str(val) + ", "
        if self.is_success is not None:
            output_string = "output: "
            for key, val in zip(get_output_header(), self.get_output_row()):
                output_string += key + "=" + str(val) + ", "
            return cfg_string.strip(", ") + "\n" + output_string.strip(", ")
        else:
            return cfg_string.strip(", ")

    def get_output_row(self):
        return [self.sample_size, self.with_ml, self.is_success, self.is_fail, self.is_abort, self.ser_completed_only, self.ser_fail_only, self.key_rate, self.key_rate_completed_only, self.key_rate_success_only, self.leak_rate, self.leak_rate_completed_only, self.leak_rate_success_only, self.matrix_size_rate, self.matrix_size_rate_success_only,
                self.bob_communication_rate, self.bob_communication_rate_success_only, self.total_communication_rate, self.total_communication_rate_success_only, self.time_rate, self.time_rate_success_only]

    def get_row(self):
        return self.get_cfg_row() + self.get_output_row()


def get_header():
    return get_cfg_header() + get_output_header()

def get_old_header():
    header = get_header()
    return header[:18] + [header[19]] + header[24:]

def get_cfg_header():
    return ["base", "N", "block_length", "num_blocks", "p_err", "success_rate", "goal_candidates_num", "max_candidates_num",
            "max_num_indices_to_encode", "predetermined_number_of_encodings", "code_strategy", "rounding_strategy", "pruning_strategy", "radius_picking", "encoding_sample_size", "theoretic_key_rate"]


def get_output_header():
    return ["sample_size", "with_ml", "is_success", "is_fail", "is_abort", "ser_completed_only", "ser_fail_only", "key_rate", "key_rate_completed_only", "key_rate_success_only",
            "leak_rate", "leak_rate_completed_only", "leak_rate_success_only", "matrix_size_rate", "matrix_size_rate_success_only", "bob_communication_rate", "bob_communication_rate_success_only", "total_communication_rate", "total_communication_rate_success_only", "time_rate", "time_rate_success_only"]

def str_to_result(row_string):
    list_of_strings = row_string[1:-1].split(", ")

    # if len(list_of_strings) == 31:
    #     # Old format -> populate the new fields
    #     list_of_strings.insert(18, "False")
    #     list_of_strings.insert(20, None)
    #     list_of_strings.insert(21, None)
    #     list_of_strings.insert(22, None)
    #     list_of_strings.insert(23, None)
    #     list_of_strings.insert(25, None)
    #     list_of_strings.insert(28, None)

    # print(list_of_strings[0])
    base = int(list_of_strings[0])
    block_length = int(list_of_strings[2])
    num_blocks = int(list_of_strings[3])
    p_err = float(list_of_strings[4])
    success_rate = float(list_of_strings[5])
    goal_candidates_num = int(list_of_strings[6])
    max_candidates_num = int(list_of_strings[7])
    max_num_indices_to_encode = int(list_of_strings[8])
    fixed_number_of_encodings = list_of_strings[9] in ["'True'", "True"]
    code_strategy = CodeStrategy[list_of_strings[10][1:-1]]
    rounding_strategy = RoundingStrategy[list_of_strings[11][1:-1]]
    pruning_strategy = PruningStrategy[list_of_strings[12][1:-1]]
    radius_picking = list_of_strings[13] in ["'True'", "True"]
    encoding_sample_size = int(list_of_strings[14])
    cfg = MbCfg(q=base, block_length=block_length, num_blocks=num_blocks,
              p_err=p_err,
              success_rate=success_rate,
              goal_candidates_num=goal_candidates_num,
              rounding_strategy=rounding_strategy,
              pruning_strategy=pruning_strategy,
              fixed_number_of_encodings=fixed_number_of_encodings,
              max_num_indices_to_encode=max_num_indices_to_encode,
              radius_picking=radius_picking,
              max_candidates_num=max_candidates_num,
              encoding_sample_size=encoding_sample_size)
    sample_size = int(list_of_strings[16])
    with_ml = metric_to_val(list_of_strings[17])
    is_success = metric_to_val(list_of_strings[18])
    is_fail = metric_to_val(list_of_strings[19])
    is_abort = metric_to_val(list_of_strings[20])
    ser_completed_only = metric_to_val(list_of_strings[21])
    ser_fail_only = metric_to_val(list_of_strings[22])
    key_rate = float(list_of_strings[23])
    key_rate_completed_only = metric_to_val(list_of_strings[24])
    key_rate_success_only = metric_to_val(list_of_strings[25])
    leak_rate = float(list_of_strings[26])
    leak_rate_completed_only = metric_to_val(list_of_strings[27])
    leak_rate_success_only = metric_to_val(list_of_strings[28])
    matrix_size_rate = float(list_of_strings[29])
    matrix_size_rate_success_only = metric_to_val(list_of_strings[30])
    bob_communication_rate = float(list_of_strings[31])
    bob_communication_rate_success_only = metric_to_val(list_of_strings[32])
    total_communication_rate = float(list_of_strings[33])
    total_communication_rate_success_only = metric_to_val(list_of_strings[34])
    time_rate = float(list_of_strings[35])
    time_rate_success_only = metric_to_val(list_of_strings[36])
    return Result(cfg=cfg, with_ml=with_ml, is_success=is_success, is_fail=is_fail, is_abort=is_abort, ser_completed_only=ser_completed_only, ser_fail_only=ser_fail_only, key_rate=key_rate, key_rate_completed_only=key_rate_completed_only, key_rate_success_only=key_rate_success_only, leak_rate=leak_rate, leak_rate_completed_only=leak_rate_completed_only, leak_rate_success_only=leak_rate_success_only,
                  matrix_size_rate=matrix_size_rate, matrix_size_rate_success_only=matrix_size_rate_success_only, bob_communication_rate=bob_communication_rate, bob_communication_rate_success_only=bob_communication_rate_success_only,
                  total_communication_rate=total_communication_rate, total_communication_rate_success_only=total_communication_rate_success_only, time_rate=time_rate, time_rate_success_only=time_rate_success_only, result_list=None, sample_size=sample_size)

def metric_to_val(metric):
    if metric in [None, "None"]:
        return None
    if metric in ["'True'", "True"]:
        return True
    if metric in ["'False'", "False"]:
        return False
    return float(metric)

def result_str_to_cfg_str(result_str):
    list_of_strings = result_str[1:-1].split(", ")
    cfg_strings = list_of_strings[:len(get_cfg_header())-1]
    return "[" + ", ".join(cfg_strings) + "]"

def result_csv_to_cfg_str(result_csv):
    cfg_csv = result_csv[:len(get_cfg_header())-1]
    return str(cfg_csv.tolist())

def convert_output_file_to_cfg_string_list(input_file_name, input_format):
    if input_format == "str":
        input_txt_file = open(input_file_name, 'r')
        input_txt_rows = input_txt_file.read().splitlines()
        assert (input_txt_rows[0] in [str(get_header()), str(get_old_header())])

        cfg_string_list = []
        for input_txt_row in input_txt_rows[1:]:
            if input_txt_row[0] != "[":
                continue
            cur_cfg_string = result_str_to_cfg_str(input_txt_row)
            cfg_string_list.append(cur_cfg_string)

        cfg_string_set = set(cfg_string_list)

        return cfg_string_set

    if input_format == "csv":
        df = pd.read_csv(input_file_name)
        assert (list(df) == get_header())

        cfg_str_list = df.apply(func=result_csv_to_cfg_str, axis=1, raw=True).values.tolist()
        cfg_str_set = set(cfg_str_list)

        return cfg_str_set

    raise "Unknown output file format"

def convert_output_glob_to_cfg_string_list(input_files_glob, input_format):
    input_file_names = glob.glob(input_files_glob)
    list_of_cfg_sets = [convert_output_file_to_cfg_string_list(input_file_name, input_format) for input_file_name in input_file_names]
    return set().union(*list_of_cfg_sets)

def convert_output_files_to_cfg_string_list(input_files_globs, input_format):
    list_of_cfg_sets = [convert_output_glob_to_cfg_string_list(input_files_glob, input_format) for input_files_glob in input_files_globs]
    return set().union(*list_of_cfg_sets)