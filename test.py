#!/usr/bin/env python3

import arg_module
import run_module
from cfg import CodeStrategy
from mb.mb_cfg import RoundingStrategy, PruningStrategy
import sys

is_slurm = False
verbosity = True
run_mode = "parallel"
p_err = float(sys.argv[1])
# p_err = 0.0

# previous_run_files = [r'/tmp/history/{p_err}/slurm-*.out'.format(p_err=p_err)]
# previous_run_file_format = "str"
previous_run_files = ["/cs/usr/benjilieber/PycharmProjects/multi_block_protocol/results/run_results_agg.csv"]
previous_run_file_format = "csv"
# previous_run_files = None
# previous_run_file_format = None

q_list = [7]
p_err_range = [p_err]  # [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1]
N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
use_log = True
sample_size = 10
# raw_results_file_path = "results/old/fake_results.csv"
# agg_results_file_path = "results/old/fake_results_agg.csv"
raw_results_file_path = "/cs/usr/benjilieber/PycharmProjects/multi_block_protocol/results/run_results.csv"
agg_results_file_path = "/cs/usr/benjilieber/PycharmProjects/multi_block_protocol/results/run_results_agg.csv"

code_strategy_list = [CodeStrategy.mb]

# Multi-block parameters
success_rate_range = [0.9, 0.99, 0.999, 0.9999]
block_size_range = list(range(3, 20))
goal_candidates_num = None
max_candidates_num = 100_000
max_num_indices_to_encode_range = [2, 4, 8, None]
fixed_number_of_encodings = False
radius_picking = False
rounding_strategy_list = [RoundingStrategy.floor]
pruning_strategy = PruningStrategy.relative_weights
encoding_sample_size = 1

# LDPC parameters
sparsity_range = None

# Polar codes parameters
constr_l = 100
relative_gap_rate_list = [0.95, 0.975, 1.0, 1.025, 1.05]
scl_l = None

args = arg_module.create_args(N_list=N_list,
                              block_size_range=block_size_range,
                              p_err_range=p_err_range,
                              success_rate_range=success_rate_range,
                              q_list=q_list,
                              code_strategy_list=code_strategy_list,
                              use_log=use_log,
                              # Running environment parameters
                              verbosity=verbosity,
                              is_slurm=is_slurm,
                              previous_run_files=previous_run_files,
                              previous_run_file_format=previous_run_file_format,
                              raw_results_file_path=raw_results_file_path,
                              agg_results_file_path=agg_results_file_path,
                              run_mode=run_mode,
                              sample_size=sample_size,
                              # Multi-block parameters
                              goal_candidates_num=goal_candidates_num,
                              max_candidates_num=max_candidates_num,
                              fixed_number_of_encodings=fixed_number_of_encodings,
                              max_num_indices_to_encode_range=max_num_indices_to_encode_range,
                              radius_picking=radius_picking,
                              rounding_strategy_list=rounding_strategy_list,
                              pruning_strategy=pruning_strategy,
                              encoding_sample_size=encoding_sample_size,
                              # LDPC parameters
                              sparsity_range=sparsity_range,
                              # Polar codes parameters
                              constr_l=constr_l,
                              relative_gap_rate_list=relative_gap_rate_list,
                              scl_l=scl_l)
run_module.multi_run(args)

# import plot
# plot.plot_results1("agg_results2.csv", q_filter=[2], qer_filter=None)
# plot.plot_results4("agg_results.csv")
# plot.plot_results4("agg_results1.csv")
# plot.plot_results4("agg_results2.csv")

# agg_results.csv, agg_results4.csv (look only at q=3) - aim for success rate + reduce when reaching threshold
# agg_results1.csv - aim for square(block_length) candidates at each iteration
# agg_results2.csv - aim for sqrt(key_length) candidates at each iteration