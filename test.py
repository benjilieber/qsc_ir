import arg_module
import run_module
from protocol_configs import CodeGenerationStrategy, RoundingStrategy, PruningStrategy

is_slurm = True
verbosity = False

p_err = 0.0
previous_run_files = [r'/tmp/history/{p_err}/slurm-*.out'.format(p_err=p_err)]
previous_run_file_format = "str"
# previous_run_files = ["/tmp/history/history_agg.csv"]
# previous_run_file_format = "txt"
previous_run_files = None
previous_run_file_format = None

block_size_range = list(range(3, 20))
key_size_list = [128, 256, 512, 1024, 2048, 4096, 8192]
p_err_range = [p_err]  # [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1]
success_rate_range = [0.9, 0.99, 0.999, 0.9999]
raw_results_file_path = "fake_results.csv"
agg_results_file_path = "fake_results_agg.csv"
run_mode = "parallel"
sample_size = 10
q_list = [3]
code_generation_strategy_list = [CodeGenerationStrategy.linear]
sparsity_range = None
goal_candidates_num = None
max_candidates_num = 100_000
max_num_indices_to_encode_range = [2, 4, 8, None]
fixed_number_of_encodings = False
radius_picking = False
rounding_strategy_list = [RoundingStrategy.floor]
pruning_strategy = PruningStrategy.relative_weights
encoding_sample_size = 1

args = arg_module.create_args(key_size_list, block_size_range, p_err_range, success_rate_range, is_slurm, previous_run_files, previous_run_file_format, raw_results_file_path, agg_results_file_path,
                              run_mode, sample_size, q_list, code_generation_strategy_list, sparsity_range, goal_candidates_num, max_candidates_num, fixed_number_of_encodings,
                              max_num_indices_to_encode_range, radius_picking, rounding_strategy_list, pruning_strategy, encoding_sample_size, verbosity)
run_module.multi_run(args)

# import plot
# plot.plot_results1("agg_results2.csv", q_filter=[2], qer_filter=None)
# plot.plot_results4("agg_results.csv")
# plot.plot_results4("agg_results1.csv")
# plot.plot_results4("agg_results2.csv")

# agg_results.csv, agg_results4.csv (look only at q=3) - aim for success rate + reduce when reaching threshold
# agg_results1.csv - aim for square(block_length) candidates at each iteration
# agg_results2.csv - aim for sqrt(key_length) candidates at each iteration