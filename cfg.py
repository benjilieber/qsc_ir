import math
from enum import Enum

class CodeStrategy(Enum):
    mb = 'mb'
    ldpc = 'ldpc'
    polar = 'polar'

    def __str__(self):
        return self.value

class Cfg(object):

    def __init__(self, orig_cfg=None, q=None, N=None, p_err=0, use_log=None,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        if orig_cfg is not None:
            self.verbosity = orig_cfg.verbosity

            self.q = orig_cfg.q
            self.N = orig_cfg.N
            self.p_err = orig_cfg.p_err

            self.use_log = orig_cfg.use_log

            self.raw_results_file_path = orig_cfg.raw_results_file_path
            self.agg_results_file_path = orig_cfg.agg_results_file_path

            self.theoretic_key_rate = orig_cfg.theoretic_key_rate

        else:
            self.verbosity = verbosity

            self.q = q
            self.N = N
            self.p_err = p_err

            self.use_log = use_log

            self.raw_results_file_path = raw_results_file_path
            self.agg_results_file_path = agg_results_file_path

            self.theoretic_key_rate = self._theoretic_key_rate()

    def _theoretic_key_rate(self):
        if self.p_err == 0.0:
            return math.log(self.q/(self.q-1), 2)
        if self.p_err == 1.0:
            return math.log(self.q, 2)
        return math.log(self.q, 2) + self.p_err * math.log(self.p_err, 2) + (1-self.p_err) * math.log((1-self.p_err)/(self.q-1), 2)

    def _theoretic_key_q_rate(self):
        return self._theoretic_key_rate() * self.log(2, self.q)

    def log_dict(self):
        specific_dict = {"q": self.q,
                         "N": self.N,
                         "p_err": self.p_err,
                         "qer": 1 - self.p_err,
                         "use_log": self.use_log,
                         "theoretic_key_rate": self.theoretic_key_rate}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return specific_dict
def specific_log_header():
    return ["q",
            "N",
            "p_err",
            "qer",
            "use_log",
            "theoretic_key_rate"]
def specific_log_header_params():
    return ["q",
            "N",
            "p_err",
            "use_log"]