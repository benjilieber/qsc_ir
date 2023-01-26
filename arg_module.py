import argparse
import math
import re

import numpy as np

from protocol_configs import CodeGenerationStrategy, RoundingStrategy, PruningStrategy


def parseIntRange(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = int(m.group(1), 10)
    end = int(m.group(2), 10) if m.group(2) else start + 1
    return list(range(start, end))


def parseFloatRange(string):
    m = re.match(r'(\d+\.\d+)(?:-(\d+\.\d+)-(\d+\.\d+))?$', string)
    if not m:
        raise argparse.ArgumentTypeError(
            "'" + string + "' is not a range of number. Expected forms like '0.1-0.5-0.1' or '0.2'.")
    start = float(m.group(1))
    if not m.group(2):
        return [start]
    end = float(m.group(2))
    step = float(m.group(3))
    return np.arange(start, end + step, step).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description='Run key generation protocol.')
    parser.add_argument('--series', dest='run_mode', action='store_const', const='series', default='parallel',
                        help='Run in series (default is parallel).')
    parser.add_argument('--sample_size', default=1, type=int, help='The number of runs per configuration.')
    parser.add_argument('--q_list', type=int, default=[], nargs="+", help='The bases of the symbol field (default: 3).')
    parser.add_argument('--code_generation_strategy_list', choices=list(CodeGenerationStrategy),
                        default=[CodeGenerationStrategy.linear], nargs="+", type=CodeGenerationStrategy,
                        help='The code types.')
    parser.add_argument('--key_size_list', required=True, nargs="+", type=int, help='The key sizes.')
    parser.add_argument('--block_size_range', required=True, type=parseIntRange, help='The block sizes.')
    parser.add_argument('--sparsity_range', type=parseIntRange, help='The LDPC code sparsity range.')
    parser.add_argument('--goal_candidates_num', type=int, nargs="+",
                        help='The max candidates list size during decoding (default: ranges over powers of 2 until key size).')
    parser.add_argument('--fixed_number_of_encodings', default=False, type=bool,
                        help='Whether the number of encodings should be fixed (otherwise based on list size requirements). By default False.')
    parser.add_argument('--max_num_indices_to_encode_range', default=[math.inf], type=int, nargs="+",
                        help='The max number of indices to encode for Linear Code.')
    parser.add_argument('--p_err_range', required=True, type=parseFloatRange, help='The error probability values.')
    parser.add_argument('--success_rate_range', type=parseFloatRange,
                        help='The success rate values.')
    parser.add_argument('--radius_picking', default=False, type=bool, help='Whether to use single-block radius-picking pruning.')
    parser.add_argument('--rounding_strategy_list', default=[RoundingStrategy.ceil], type=list(RoundingStrategy), help='Which rounding function to use (for encoding-number picking).')
    parser.add_argument('--pruning_strategy', type=PruningStrategy, help='The pruning strategy.')
    parser.add_argument('--max_candidates_num', type=int, help='The upper threshold for the list size, from which the list should be reduced when reached.')
    parser.add_argument('--encoding_sample_size', type=int, help='The number of encodings sampled to pick the best one.')
    parser.add_argument('--is_slurm', type=bool, required=True, help='Whether we are running on slurm.')
    parser.add_argument('--previous_run_files', required=False, type=str,
                        help='Paths of raw results of previous runs. If they exist, the new run will start where the previous ones left off.')
    parser.add_argument('--previous_run_file_format', default="str", required=False, type=str,
                        help='Format of raw results of previous run. Default is sample text.')
    parser.add_argument('--raw_results_file_path', required=True, type=str,
                        help='csv path for saving the raw results. If exists, the new results are appended to the existing ones.')
    parser.add_argument('--agg_results_file_path', required=True, type=str,
                        help='csv path for saving the aggregated results. If exists, the new results are appended to the existing ones.')
    parser.add_argument("--verbosity", default=False, type=bool, help='Verbosity.')

    args = parser.parse_args()
    if CodeGenerationStrategy.ldpc in args.code_generation_strategy_list:
        assert args.sparsity_range is not None

    return args


def create_args(key_size_list, block_size_range, p_err_range, success_rate_range, is_slurm, previous_run_files, previous_run_file_format, raw_results_file_path,
                agg_results_file_path, run_mode='series', sample_size=1, q_list=[3],
                code_generation_strategy_list=[CodeGenerationStrategy.linear], sparsity_range=None,
                goal_candidates_num=None, max_candidates_num=None, fixed_number_of_encodings=False,
                max_num_indices_to_encode_range=[math.inf], radius_picking=False, rounding_strategy_list=[RoundingStrategy.ceil], pruning_strategy=PruningStrategy.radii_probabilities, encoding_sample_size=1, verbosity=False):
    class Args(object):
        def __init__(self, key_size_list, block_size_range, p_err_range, success_rate_range, is_slurm, previous_run_files, previous_run_file_format, raw_results_file_path,
                     agg_results_file_path, run_mode='series', sample_size=1, q_list=3,
                     code_generation_strategy_list=[CodeGenerationStrategy.linear], sparsity_range=None,
                     goal_candidates_num=None, max_candidates_num=None, fixed_number_of_encodings=False,
                     max_num_indices_to_encode_range=[math.inf], radius_picking=False, rounding_strategy_list=[RoundingStrategy.ceil], pruning_strategy=PruningStrategy.radii_probabilities, encoding_sample_size=1, verbosity=False):
            self.run_mode = run_mode
            self.sample_size = sample_size
            self.q_list = q_list
            self.code_generation_strategy_list = code_generation_strategy_list
            self.sparsity_range = sparsity_range
            if CodeGenerationStrategy.ldpc in self.code_generation_strategy_list:
                assert self.sparsity_range is not None
            self.key_size_list = key_size_list
            self.block_size_range = block_size_range
            self.goal_candidates_num = goal_candidates_num
            self.max_candidates_num = max_candidates_num
            self.fixed_number_of_encodings = fixed_number_of_encodings
            self.max_num_indices_to_encode_range = max_num_indices_to_encode_range
            self.p_err_range = p_err_range
            self.success_rate_range = success_rate_range
            self.radius_picking = radius_picking
            self.rounding_strategy_list = rounding_strategy_list
            self.pruning_strategy = pruning_strategy
            self.encoding_sample_size = encoding_sample_size
            self.is_slurm = is_slurm
            self.previous_run_files = previous_run_files
            self.previous_run_file_format = previous_run_file_format
            self.raw_results_file_path = raw_results_file_path
            self.agg_results_file_path = agg_results_file_path
            self.verbosity = verbosity

    return Args(key_size_list, block_size_range, p_err_range, success_rate_range, is_slurm, previous_run_files, previous_run_file_format, raw_results_file_path, agg_results_file_path,
                run_mode, sample_size, q_list, code_generation_strategy_list, sparsity_range, goal_candidates_num, max_candidates_num, fixed_number_of_encodings,
                max_num_indices_to_encode_range, radius_picking, rounding_strategy_list, pruning_strategy, encoding_sample_size, verbosity)

