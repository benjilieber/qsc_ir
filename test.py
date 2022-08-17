import arg_module
import run_module
from protocol_configs import CodeGenerationStrategy, PruningStrategy

block_size = 25
num_blocks = 10
block_size_range = [block_size]
key_size_list = [num_blocks*block_size]
p_err_range = [0.11]
success_rate_range = None
raw_results_file_path = "raw_results5.csv"
agg_results_file_path = "agg_results5.csv"
run_mode = "series"
sample_size = 5
q = 2
hash_base_list = [2]
code_generation_strategy_list = [CodeGenerationStrategy.LINEAR_CODE]
sparsity_range = None
max_candidates_num = None  # 100
max_num_indices_to_encode_range = [4]
fixed_number_of_encodings = True
radius = None
pruning_strategy = PruningStrategy.RELATIVE_WEIGHTS
upper_threshold = 10_000
cfg_timeout = None
verbosity = True

args = arg_module.create_args(key_size_list, block_size_range, p_err_range, success_rate_range, raw_results_file_path, agg_results_file_path,
                run_mode, sample_size, q, hash_base_list, code_generation_strategy_list, sparsity_range, max_candidates_num, fixed_number_of_encodings,
                max_num_indices_to_encode_range, radius, pruning_strategy, upper_threshold, cfg_timeout, verbosity)
run_module.multi_run(args)