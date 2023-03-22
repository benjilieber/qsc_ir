import pandas as pd

import result
from cfg import CodeStrategy
from mb.mb_cfg import IndicesToEncodeStrategy

old_file_name = "results/mb,q=5,agg.csv"
new_file_name = "results/mb,q=5,agg-new.csv"
old_df = pd.read_csv(old_file_name)

old_df = old_df[old_df.p_err.isin([0.0, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1])]
old_df = old_df[old_df.with_ml.isin([False, True])]
# print((~old_df.with_ml.astype(bool)).unique())
# print((~old_df.with_ml.astype(bool)).value_counts())

new_df = pd.DataFrame(index=range(old_df.shape[0]), columns=result.get_header())
new_df.q = old_df.base.astype(int)
new_df.N = old_df.N.astype(int)
new_df.p_err = old_df.p_err
new_df.qer = 1.0-old_df.p_err.astype(float)
new_df.use_log = True
new_df.list_size = old_df.goal_candidates_num.astype(int)
new_df.check_length = 0

new_df.theoretic_key_rate = old_df.theoretic_key_rate
new_df.code_strategy = old_df.code_strategy
new_df.mb_desired_success_rate = old_df.success_rate
new_df.mb_block_length = old_df.block_length.astype(int)
new_df.mb_num_blocks = old_df.num_blocks.astype(int)
new_df.mb_full_rank_encoding = True
new_df.mb_use_zeroes_in_encoding_matrix = True
new_df.mb_indices_to_encode_strategy = str(IndicesToEncodeStrategy.most_candidate_blocks)
new_df.mb_rounding_strategy = old_df.rounding_strategy
new_df.mb_pruning_strategy = old_df.pruning_strategy
new_df.mb_max_num_indices_to_encode = old_df.max_num_indices_to_encode.astype(int)
new_df.mb_max_candidates_num = old_df.max_candidates_num.astype(int)
new_df.mb_encoding_sample_size = old_df.encoding_sample_size.astype(int)
new_df.mb_radius_picking = old_df.radius_picking
new_df.mb_predetermined_number_of_encodings = old_df.predetermined_number_of_encodings
new_df.sample_size = old_df.sample_size.astype(int)
old_df["result_type"] = ""
old_df.loc[old_df.with_ml.astype(bool), "result_type"] = str(result.ResultType.full_list)
old_df.loc[~old_df.with_ml.astype(bool), "result_type"] = str(result.ResultType.reduced)
new_df.result_type = old_df.result_type
new_df.is_success = old_df.is_success
new_df.ser_b_key_completed_only = old_df.ser_completed_only
new_df.ser_b_key_fail_only = old_df.ser_fail_only
new_df.ser_a_guess_completed_only = old_df.ser_completed_only
new_df.ser_a_guess_fail_only = old_df.ser_fail_only
new_df.key_rate = old_df.key_rate
new_df.key_rate_completed_only = old_df.key_rate_completed_only
new_df.key_rate_success_only = old_df.key_rate_success_only
new_df.leak_rate = old_df.leak_rate
new_df.leak_rate_completed_only = old_df.leak_rate_completed_only
new_df.leak_rate_success_only = old_df.leak_rate_success_only
new_df.matrix_size_rate = old_df.matrix_size_rate
new_df.matrix_size_rate_success_only = old_df.matrix_size_rate_success_only
new_df.bob_communication_rate = old_df.bob_communication_rate
new_df.bob_communication_rate_success_only = old_df.bob_communication_rate_success_only
new_df.total_communication_rate = old_df.total_communication_rate
new_df.total_communication_rate_success_only = old_df.total_communication_rate_success_only
new_df.time_rate = old_df.time_rate
new_df.time_rate_success_only = old_df.time_rate_success_only

new_df.to_csv(new_file_name, index=False)