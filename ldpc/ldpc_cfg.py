import math
from enum import Enum

from cfg import Cfg, CodeStrategy

class Decoder(Enum):
    bp = 'bp'
    it = 'it'

    def __str__(self):
        return self.value

class LdpcCfg(Cfg):

    def __init__(self,
                 orig_cfg=None,
                 q=None,
                 N=None,
                 p_err=0,
                 key_rate=None,
                 syndrome_length=None,
                 success_rate=None,
                 relative_gap_rate=None,
                 sparsity=1.0,
                 decoder=None,
                 max_num_rounds=None,
                 L=None,
                 use_forking=None,
                 use_hints=None,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        super().__init__(orig_cfg=orig_cfg, q=q, N=N, p_err=p_err, code_strategy=CodeStrategy.ldpc, raw_results_file_path=raw_results_file_path,
                         agg_results_file_path=agg_results_file_path, verbosity=verbosity)

        assert ((key_rate is not None) + (syndrome_length is not None) + (success_rate is not None) + (relative_gap_rate is not None) == 1)
        if key_rate is not None:
            self.syndrome_length = int(math.ceil(key_rate * math.log(2, self.q) * self.N))
        elif syndrome_length is not None:
            self.syndrome_length = int(syndrome_length)
        elif success_rate is not None:
            raise "No support for ldpc success rate input yet"
        elif relative_gap_rate is not None:
            self.syndrome_length = int(math.ceil(self._theoretic_key_q_rate() * relative_gap_rate * self.N))
        self.desired_key_rate = key_rate
        self.desired_syndrome_length = syndrome_length
        self.desired_success_rate = success_rate
        self.desired_relative_gap_rate = relative_gap_rate

        self.sparsity = sparsity

        self.decoder = decoder
        self.max_num_rounds = max_num_rounds
        self.L = L
        self.use_forking = use_forking
        self.use_hints = use_hints

    def log_dict(self):
        super_dict = super().log_dict()
        specific_dict = {"ldpc_sparsity": self.sparsity,
                         "ldpc_syndrome_length": self.syndrome_length,
                         "ldpc_desired_key_rate": self.desired_key_rate,
                         "ldpc_desired_syndrome_length": self.desired_syndrome_length,
                         "ldpc_desired_success_rate": self.desired_success_rate,
                         "ldpc_desired_relative_gap_rate": self.desired_relative_gap_rate,
                         "ldpc_decoder": self.decoder,
                         "ldpc_max_num_rounds": self.max_num_rounds,
                         "ldpc_L": self.L,
                         "ldpc_use_forking": self.use_forking,
                         "ldpc_use_hints": self.use_hints}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return {**super_dict, **specific_dict}


def specific_log_header():
    return ["ldpc_syndrome_length",
            "ldpc_sparsity",
            "ldpc_decoder",
            "ldpc_max_num_rounds",
            "ldpc_L",
            "ldpc_use_forking",
            "ldpc_use_hints"]


def specific_log_header_params():
    return ["ldpc_syndrome_length",
            "ldpc_sparsity",
            "ldpc_decoder",
            "ldpc_max_num_rounds",
            "ldpc_L",
            "ldpc_use_forking",
            "ldpc_use_hints"]
