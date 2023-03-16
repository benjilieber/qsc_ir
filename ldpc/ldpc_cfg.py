from cfg import Cfg, CodeStrategy


class LdpcCfg(Cfg):

    def __init__(self, orig_cfg=None, q=None, N=None, p_err=0, sparsity=1.0,
                 raw_results_file_path=None,
                 agg_results_file_path=None,
                 verbosity=False):
        super().__init__(orig_cfg=orig_cfg, q=q, N=N, p_err=p_err, code_strategy=CodeStrategy.ldpc, raw_results_file_path=raw_results_file_path,
                         agg_results_file_path=agg_results_file_path, verbosity=verbosity)

        self.sparsity = sparsity

    def log_dict(self):
        super_dict = super().log_dict()
        specific_dict = {"ldpc_sparsity": self.sparsity}
        assert (set(specific_dict.keys()) == set(specific_log_header()))
        return {**super_dict, **specific_dict}


def specific_log_header():
    return ["ldpc_sparsity"]


def specific_log_header_params():
    return ["ldpc_sparsity"]
