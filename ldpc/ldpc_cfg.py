from cfg import Cfg, CodeStrategy

class LdpcCfg(Cfg):

    def __init__(self, orig_cfg=None, q=None, N=None, p_err=0, sparsity=1.0,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        self.code_strategy = CodeStrategy.ldpc
        super().__init__(orig_cfg=orig_cfg, q=q, N=N, p_err=p_err, raw_results_file_path=raw_results_file_path, agg_results_file_path=agg_results_file_path, verbosity=verbosity)

        self.sparsity = sparsity

    def log_dict(self):
        super_dict = super().log_dict()
        specific_dict = {"code_strategy": str(self.code_strategy),
                         "ldpc_sparsity": self.sparsity}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return {**super_dict, **specific_dict}

def specific_log_header():
    return ["code_strategy",
            "ldpc_sparsity"]

def specific_log_header_params():
    return ["code_strategy",
            "ldpc_sparsity"]