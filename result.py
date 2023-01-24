import numpy as np


class Result(object):
    def __init__(self, cfg, is_success=None, key_rate=None, encoding_size_rate=None, matrix_size_rate=None, bob_communication_rate=None,
                 total_communication_rate=None, time_rate=None, result_list=None):
        self.cfg = cfg
        if result_list is None:
            self.is_success = is_success
            self.key_rate = key_rate
            self.key_rate_success_only = key_rate
            self.encoding_size_rate = encoding_size_rate
            self.encoding_size_rate_success_only = encoding_size_rate
            self.matrix_size_rate = matrix_size_rate
            self.matrix_size_rate_success_only = matrix_size_rate
            self.bob_communication_rate = bob_communication_rate
            self.bob_communication_rate_success_only = bob_communication_rate
            self.total_communication_rate = total_communication_rate
            self.total_communication_rate_success_only = total_communication_rate
            self.time_rate = time_rate
            self.time_rate_success_only = time_rate
            self.sample_size = 1
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
