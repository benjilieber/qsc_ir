import arg_module
import plot
import run_module
from protocol_configs import CodeGenerationStrategy, RoundingStrategy, PruningStrategy

is_slurm = True
verbosity = True

block_size_range = [8]  # list(range(3, 20))
key_size_list = [1000]
p_err_range = [0.01]  # [0.0, 0.01, 0.02, 0.05, 0.11]
success_rate_range = None
raw_results_file_path = "raw_results1.csv"
agg_results_file_path = "agg_results1.csv"
run_mode = "series"
sample_size = 10
q_list = [3]
code_generation_strategy_list = [CodeGenerationStrategy.LINEAR_CODE]
sparsity_range = None
goal_candidates_num = 1000
max_candidates_num = 10_000
max_num_indices_to_encode_range = [None]  # [1, 2, 4, 8, None]
fixed_number_of_encodings = True
radius_picking = False
rounding_strategy_list = [RoundingStrategy.FLOOR, RoundingStrategy.CEIL]
pruning_strategy = PruningStrategy.RELATIVE_WEIGHTS
encoding_sample_size = 1

args = arg_module.create_args(key_size_list, block_size_range, p_err_range, success_rate_range, is_slurm, raw_results_file_path, agg_results_file_path,
                              run_mode, sample_size, q_list, code_generation_strategy_list, sparsity_range, goal_candidates_num, max_candidates_num, fixed_number_of_encodings,
                              max_num_indices_to_encode_range, radius_picking, rounding_strategy_list, pruning_strategy, encoding_sample_size, verbosity)
run_module.multi_run(args)

# plot.plot_results1("agg_results2.csv", q_filter=[2], qer_filter=None)
# plot.plot_results4("agg_results.csv")
# plot.plot_results4("agg_results1.csv")
# plot.plot_results4("agg_results2.csv")

# agg_results.csv, agg_results4.csv (look only at q=3) - aim for success rate + reduce when reaching threshold
# agg_results1.csv - aim for square(block_length) candidates at each iteration
# agg_results2.csv - aim for sqrt(key_length) candidates at each iteration