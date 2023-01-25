import numpy as np

from protocol_configs import CodeGenerationStrategy, PruningStrategy, RoundingStrategy, ProtocolConfigs


class Result(object):
    def __init__(self, cfg, is_success=None, key_rate=None, key_rate_success_only=None, encoding_size_rate=None, encoding_size_rate_success_only=None, matrix_size_rate=None, matrix_size_rate_success_only=None, bob_communication_rate=None,
                 bob_communication_rate_success_only=None, total_communication_rate=None, total_communication_rate_success_only=None, time_rate=None, time_rate_success_only=None, result_list=None, sample_size=None):
        self.cfg = cfg
        if result_list is None:
            self.is_success = is_success
            self.key_rate = key_rate
            self.key_rate_success_only = key_rate_success_only or key_rate
            self.encoding_size_rate = encoding_size_rate
            self.encoding_size_rate_success_only = encoding_size_rate_success_only or encoding_size_rate
            self.matrix_size_rate = matrix_size_rate
            self.matrix_size_rate_success_only = matrix_size_rate_success_only or matrix_size_rate
            self.bob_communication_rate = bob_communication_rate
            self.bob_communication_rate_success_only = bob_communication_rate_success_only or bob_communication_rate
            self.total_communication_rate = total_communication_rate
            self.total_communication_rate_success_only = total_communication_rate_success_only or total_communication_rate
            self.time_rate = time_rate
            self.time_rate_success_only = time_rate_success_only or time_rate
            self.sample_size = sample_size or 1
        else:
            assert ([cfg == result_list[i].cfg for i in range(len(result_list))])

            self.is_success = np.mean([result.is_success for result in result_list])
            success_result_list = [result for result in result_list if result.is_success]
            self.key_rate = np.mean([result.key_rate for result in result_list])
            self.key_rate_success_only = np.mean([result.key_rate for result in success_result_list])
            self.encoding_size_rate = np.mean([result.encoding_size_rate for result in result_list])
            self.encoding_size_rate_success_only = np.mean([result.encoding_size_rate for result in success_result_list])
            self.matrix_size_rate = np.mean([result.matrix_size_rate for result in result_list])
            self.matrix_size_rate_success_only = np.mean([result.matrix_size_rate for result in success_result_list])
            self.bob_communication_rate = np.mean([result.bob_communication_rate for result in result_list])
            self.bob_communication_rate_success_only = np.mean([result.bob_communication_rate for result in success_result_list])
            self.total_communication_rate = np.mean([result.total_communication_rate for result in result_list])
            self.total_communication_rate_success_only = np.mean([result.total_communication_rate for result in success_result_list])
            self.time_rate = np.mean([result.time_rate for result in result_list])
            self.time_rate_success_only = np.mean([result.time_rate for result in success_result_list])
            self.sample_size = len(result_list)

    def get_cfg_row(self):
        return [self.cfg.base, self.cfg.key_length, self.cfg.block_length, self.cfg.num_blocks, self.cfg.p_err, self.cfg.success_rate,
                self.cfg.goal_candidates_num, self.cfg.max_candidates_num,
                self.cfg.max_num_indices_to_encode, self.cfg.fixed_number_of_encodings, str(self.cfg.code_generation_strategy), str(self.cfg.rounding_strategy), str(self.cfg.pruning_strategy), self.cfg.radius_picking, self.cfg.encoding_sample_size, self.cfg.sparsity, self.cfg.theoretic_key_rate]

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
        return [self.sample_size, self.is_success, self.key_rate, self.key_rate_success_only, self.encoding_size_rate, self.encoding_size_rate_success_only, self.matrix_size_rate, self.matrix_size_rate_success_only,
                self.bob_communication_rate, self.bob_communication_rate_success_only, self.total_communication_rate, self.total_communication_rate_success_only, self.time_rate, self.time_rate_success_only]

    def get_row(self):
        return self.get_cfg_row() + self.get_output_row()


def get_header():
    return get_cfg_header() + get_output_header()

def get_cfg_header():
    return ["base", "key_length", "block_length", "num_blocks", "p_err", "success_rate", "goal_candidates_num", "max_candidates_num",
            "max_num_indices_to_encode", "predetermined_number_of_encodings", "code_generation_strategy", "rounding_strategy", "pruning_strategy", "radius_picking", "encoding_sample_size", "sparsity", "theoretic_key_rate"]

def get_output_header():
    return ["sample_size", "is_success", "key_rate", "key_rate_success_only",
            "encoding_size_rate", "encoding_size_rate_success_only", "matrix_size_rate", "matrix_size_rate_success_only", "bob_communication_rate", "bob_communication_rate_success_only", "total_communication_rate", "total_communication_rate_success_only", "time_rate", "time_rate_success_only"]

def str_to_result(row_string):
    list_of_strings = row_string[1:-1].split(", ")

    base = int(list_of_strings[0])
    block_length = int(list_of_strings[2])
    num_blocks = int(list_of_strings[3])
    p_err = float(list_of_strings[4])
    success_rate = float(list_of_strings[5])
    goal_candidates_num = int(list_of_strings[6])
    max_candidates_num = int(list_of_strings[7])
    max_num_indices_to_encode = int(list_of_strings[8])
    fixed_number_of_encodings = list_of_strings[9] in ["'True'", "True"]
    code_generation_strategy = CodeGenerationStrategy[list_of_strings[10][1:-1]]
    rounding_strategy = RoundingStrategy[list_of_strings[11][1:-1]]
    pruning_strategy = PruningStrategy[list_of_strings[12][1:-1]]
    radius_picking = list_of_strings[13] in ["'True'", "True"]
    encoding_sample_size = int(list_of_strings[14])
    sparsity = int(list_of_strings[15])
    cfg = ProtocolConfigs(base=base, block_length=block_length, num_blocks=num_blocks,
                                                              p_err=p_err,
                                                              success_rate=success_rate,
                                                              goal_candidates_num=goal_candidates_num,
                                                              rounding_strategy=rounding_strategy,
                                                              code_generation_strategy=code_generation_strategy,
                                                              pruning_strategy=pruning_strategy,
                                                              sparsity=sparsity,
                                                              fixed_number_of_encodings=fixed_number_of_encodings,
                                                              max_num_indices_to_encode=max_num_indices_to_encode,
                                                              radius_picking=radius_picking,
                                                              max_candidates_num=max_candidates_num,
                                                              encoding_sample_size=encoding_sample_size)
    sample_size = int(list_of_strings[17])
    is_success = (list_of_strings[18] in ["'True'", "True"]) if (list_of_strings[18] in ["'True'", "True", "'False'", "False"]) else float(list_of_strings[18])
    key_rate = float(list_of_strings[19])
    key_rate_success_only = float(list_of_strings[20])
    encoding_size_rate = float(list_of_strings[21])
    encoding_size_rate_success_only = float(list_of_strings[22])
    matrix_size_rate = float(list_of_strings[23])
    matrix_size_rate_success_only = float(list_of_strings[24])
    bob_communication_rate = float(list_of_strings[25])
    bob_communication_rate_success_only = float(list_of_strings[26])
    total_communication_rate = float(list_of_strings[27])
    total_communication_rate_success_only = float(list_of_strings[28])
    time_rate = float(list_of_strings[29])
    time_rate_success_only = float(list_of_strings[30])
    return Result(cfg=cfg, is_success=is_success, key_rate=key_rate, key_rate_success_only=key_rate_success_only, encoding_size_rate=encoding_size_rate, encoding_size_rate_success_only=encoding_size_rate_success_only,
                  matrix_size_rate=matrix_size_rate, matrix_size_rate_success_only=matrix_size_rate_success_only, bob_communication_rate=bob_communication_rate, bob_communication_rate_success_only=bob_communication_rate_success_only,
                 total_communication_rate=total_communication_rate, total_communication_rate_success_only=total_communication_rate_success_only, time_rate=time_rate, time_rate_success_only=time_rate_success_only, result_list=None, sample_size=sample_size)