#!/usr/bin/env python3
import sys

import arg_module
import run_module
from cfg import CodeStrategy
from ldpc.ldpc_cfg import Decoder
from mb.mb_cfg import RoundingStrategy, PruningStrategy

verbosity = False
run_mode = "parallel"
p_err = float(sys.argv[1])
code_strategy = CodeStrategy[sys.argv[2]]

# code_strategy = CodeStrategy.polar  # [CodeStrategy.mb, CodeStrategy.polar, CodeStrategy.ldpc]
code_strategy_list = [code_strategy]
q = 3
q_list = [q]
p_err_range = [p_err]  # [0.0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1]
N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
use_log = True
sample_size = 100
# raw_results_file_path = "results/old/fake_results.csv"
# agg_results_file_path = "results/old/fake_results_agg.csv"
raw_results_file_path = "results/{code_strategy},q={q},p_err={p_err}.csv".format(code_strategy=str(code_strategy), q=q, p_err=p_err)
agg_results_file_path = "results/{code_strategy},q={q},p_err={p_err},agg.csv".format(code_strategy=str(code_strategy), q=q, p_err=p_err)
print(raw_results_file_path)

# previous_run_files = [r'/tmp/history/{p_err}/slurm-*.out'.format(p_err=p_err)]
# previous_run_file_format = "str"
previous_run_files = [agg_results_file_path]  # ["results/mb,q=5,agg.csv"]
previous_run_file_format = "csv"
# previous_run_files = None
# previous_run_file_format = None


# Multi-block parameters
mb_success_rate_range = [0.9, 0.99, 0.999, 0.9999]
mb_block_size_range = [3, 11, 19]  # list(range(3, 20))
mb_goal_candidates_num = None
mb_max_candidates_num = 100_000
mb_max_num_indices_to_encode_range = [4, 8]  # [2, 4, 8, None]
mb_fixed_number_of_encodings = False
mb_radius_picking = False
mb_rounding_strategy_list = [RoundingStrategy.floor]
mb_pruning_strategy = PruningStrategy.relative_weights
mb_encoding_sample_size = 1

# LDPC parameters
ldpc_key_rate_list = [None]
ldpc_syndrome_length_list = [None]
ldpc_success_rate_list = [None]
ldpc_relative_gap_rate_list = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
ldpc_sparsity_range = [2, 3, 4, 5]
ldpc_decoder_list = [Decoder.bp]
ldpc_max_num_rounds_list = [30]
ldpc_L_list = [1]
ldpc_use_forking_list = [False]
ldpc_use_hints_list = [False]

# Polar codes parameters
polar_constr_l = 50
polar_key_rate_list = None
polar_num_info_indices_list = None
polar_success_rate_list = None
polar_relative_gap_rate_list = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
polar_scl_l_list = [1, 3, 9, 27, 81, 243, 729, 2187, 6561]  # , 19683

args = arg_module.create_args(N_list=N_list,
                              p_err_range=p_err_range,
                              q_list=q_list,
                              code_strategy_list=code_strategy_list,
                              use_log=use_log,
                              # Running environment parameters
                              verbosity=verbosity,
                              previous_run_files=previous_run_files,
                              previous_run_file_format=previous_run_file_format,
                              raw_results_file_path=raw_results_file_path,
                              agg_results_file_path=agg_results_file_path,
                              run_mode=run_mode,
                              sample_size=sample_size,
                              # Multi-block parameters
                              mb_success_rate_range=mb_success_rate_range,
                              mb_block_size_range=mb_block_size_range,
                              mb_goal_candidates_num=mb_goal_candidates_num,
                              mb_max_candidates_num=mb_max_candidates_num,
                              mb_fixed_number_of_encodings=mb_fixed_number_of_encodings,
                              mb_max_num_indices_to_encode_range=mb_max_num_indices_to_encode_range,
                              mb_radius_picking=mb_radius_picking,
                              mb_rounding_strategy_list=mb_rounding_strategy_list,
                              mb_pruning_strategy=mb_pruning_strategy,
                              mb_encoding_sample_size=mb_encoding_sample_size,
                              # LDPC parameters
                              ldpc_key_rate_list=ldpc_key_rate_list,
                              ldpc_syndrome_length_list=ldpc_syndrome_length_list,
                              ldpc_success_rate_list=ldpc_success_rate_list,
                              ldpc_relative_gap_rate_list=ldpc_relative_gap_rate_list,
                              ldpc_sparsity_range=ldpc_sparsity_range,
                              ldpc_decoder_list=ldpc_decoder_list,
                              ldpc_max_num_rounds_list=ldpc_max_num_rounds_list,
                              ldpc_L_list=ldpc_L_list,
                              ldpc_use_forking_list=ldpc_use_forking_list,
                              ldpc_use_hints_list=ldpc_use_hints_list,
                              # Polar codes parameters
                              polar_constr_l=polar_constr_l,
                              polar_key_rate_list=polar_key_rate_list,
                              polar_num_info_indices_list=polar_num_info_indices_list,
                              polar_success_rate_list=polar_success_rate_list,
                              polar_relative_gap_rate_list=polar_relative_gap_rate_list,
                              polar_scl_l_list=polar_scl_l_list)
run_module.multi_run(args)

# import plot
# plot.plot_results1("agg_results2.csv", q_filter=[2], qer_filter=None)
# plot.plot_results4("agg_results.csv")
# plot.plot_results4("agg_results1.csv")
# plot.plot_results4("agg_results2.csv")

# agg_results.csv, agg_results4.csv (look only at q=3) - aim for success rate + reduce when reaching threshold
# agg_results1.csv - aim for square(block_length) candidates at each iteration
# agg_results2.csv - aim for sqrt(key_length) candidates at each iteration
