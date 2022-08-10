import arg_module
import run_module
from protocol_configs import CodeGenerationStrategy

block_size = 10
num_blocks = 10
block_size_range = [block_size]
key_size_list = [num_blocks*block_size]
p_err_range = [0.02]
success_rate_range = [0.9]
raw_results_file_path = "raw_results2.csv"
agg_results_file_path = "agg_results2.csv"
run_mode = "series"
sample_size = 100
q = 3
hash_base_list = [3]
code_generation_strategy_list = [CodeGenerationStrategy.LINEAR_CODE]
sparsity_range = None
max_candidates_num = None  # 100
max_num_indices_to_encode_range = [2]
fixed_number_of_encodings = True
radius = None
cfg_timeout = None
verbosity = True

args = arg_module.create_args(key_size_list, block_size_range, p_err_range, success_rate_range, raw_results_file_path, agg_results_file_path,
                run_mode, sample_size, q, hash_base_list, code_generation_strategy_list, sparsity_range, max_candidates_num, fixed_number_of_encodings,
                max_num_indices_to_encode_range, radius, cfg_timeout, verbosity)
run_module.multi_run(args)