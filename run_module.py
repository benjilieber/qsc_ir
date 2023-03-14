import csv
import os
import sys

import pandas as pd

import cfg
import result
from ldpc import ldpc_cfg
from ldpc.ldpc_cfg import LdpcCfg
from mb import mb_cfg
from polar import polar_cfg
from polar.polar_cfg import PolarCfg

sys.path.append(os.getcwd())
import numpy as np
import time
import multiprocessing

from timeit import default_timer as timer
from mb.mb_cfg import IndicesToEncodeStrategy
from mb.mb_cfg import MbCfg
from cfg import Cfg, CodeStrategy
from key_generator import KeyGenerator
from mb.mb_protocol import MbProtocol
from result import Result
import math


def write_header(file_name):
    try:
        with open(file_name, 'r') as f:
            for row in f:
                assert (row.rstrip('\n').split(",") == result.get_header())
                return
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result.get_header())
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")


def write_results(result_pair_list, is_slurm=False, verbosity=False):
    non_ml_result_list = [result_pair[0] for result_pair in result_pair_list]
    ml_result_list = [result_pair[1] for result_pair in result_pair_list]
    if verbosity:
        print("writing results")
    if is_slurm:
        for single_result_pair in result_pair_list:
            print(single_result_pair[0].get_row())
            print(single_result_pair[1].get_row())
        print(Result(non_ml_result_list[0].cfg, with_ml=False, result_list=non_ml_result_list).get_row(), flush=True)
        print(Result(ml_result_list[0].cfg, with_ml=True, result_list=ml_result_list).get_row(), flush=True)
        return
    raw_results_file_name = result_pair_list[0][0].cfg.raw_results_file_path
    with open(raw_results_file_name, 'a', newline='') as f1:
        writer = csv.writer(f1)
        for single_result_pair in result_pair_list:
            writer.writerow(single_result_pair[0].get_row())
            writer.writerow(single_result_pair[1].get_row())

    agg_results_file_name = result_pair_list[0][0].cfg.agg_results_file_path
    with open(agg_results_file_name, 'a', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(Result(non_ml_result_list[0].cfg, with_ml=False, result_list=non_ml_result_list).get_row())
        writer.writerow(Result(ml_result_list[0].cfg, with_ml=True, result_list=ml_result_list).get_row())


def single_run(cfg):
    # print(f"Started process {os.getpid()}", flush=True)
    np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    key_generator = KeyGenerator(p_err=cfg.p_err, N=cfg.N, base=cfg.q)
    a, b = key_generator.generate_keys()
    protocol = MbProtocol(cfg, a, b)
    result_pair = protocol.run()
    # print(f"Ended process {os.getpid()}", flush=True)
    return result_pair


def multi_run_series(cfg, sample_size, is_slurm, verbosity=False):
    result_pair_list = []
    for single_sample_run in range(sample_size):
        result_pair = single_run(cfg)
        if verbosity:
            print(result_pair)
        result_pair_list.append(result_pair)
    if verbosity:
        non_ml_result_list = [result_pair[0] for result_pair in result_pair_list]
        ml_result_list = [result_pair[1] for result_pair in result_pair_list]
        print(Result(cfg, result_list=non_ml_result_list))
        print(Result(cfg, result_list=ml_result_list))
    write_results(result_pair_list, is_slurm=is_slurm, verbosity=verbosity)


def multi_run_parallel(cfg, sample_size, is_slurm, verbosity=False):
    values = [cfg] * sample_size
    # partial_write_results = partial(write_results, is_slurm=is_slurm, verbosity=verbosity)
    with multiprocessing.Pool(sample_size) as pool:
        # with multiprocessing.Pool() as pool:
        write_results(pool.map(single_run, values), is_slurm=is_slurm, verbosity=verbosity)
        if verbosity:
            print("starting to run")
        # return pool.map_async(single_run, values, callback=partial_write_results)


def multi_run(args):
    if args.is_slurm:
        print(result.get_header())
    else:
        write_header(args.raw_results_file_path)
        write_header(args.agg_results_file_path)

    start = timer()
    if args.run_mode == 'parallel':
        if args.verbosity:
            print(f'starting computations on {multiprocessing.cpu_count()} cores')
        r_list = []
    else:
        if args.verbosity:
            print('starting computations in series')

    if args.previous_run_files is not None:
        assert (len(args.previous_run_files) <= 1)
        previous_cfg_df = pd.read_csv(args.previous_run_files[0])

    for code_strategy in args.code_strategy_list:
        for q in args.q_list:
            for p_err in args.p_err_range:
                for N in args.N_list:
                    cfg = Cfg(q=q, N=N, p_err=p_err, use_log=args.use_log,
                              raw_results_file_path=args.raw_results_file_path,
                              agg_results_file_path=args.agg_results_file_path, verbosity=args.verbosity)
                    for run_cfg in generate_cfg(code_strategy, args, cfg):
                        if args.previous_run_files is not None and previous_cfg_df.apply(check_cfg_equality(run_cfg),
                                                                                         axis=1).any(axis=None):
                            if args.verbosity:
                                print("skip cfg (previously run)")
                            continue
                        if args.verbosity:
                            print(result.Result(run_cfg))
                        if args.run_mode == 'parallel':
                            # r_list.append(multi_run_parallel(cfg, args.sample_size))
                            multi_run_parallel(run_cfg, args.sample_size, is_slurm=args.is_slurm,
                                               verbosity=args.verbosity)
                        else:
                            multi_run_series(run_cfg, args.sample_size, is_slurm=args.is_slurm,
                                             verbosity=args.verbosity)

    # print("started all runs")
    # if args.run_mode == 'parallel':
    #     for r in r_list:
    #         print(r)
    #         print("wait")
    #         print(r.ready())
    #         print(r.get())
    #         # r.wait()
    end = timer()
    if args.verbosity:
        print(f'elapsed time: {end - start}')


def generate_cfg(code_strategy, args, cfg):
    if code_strategy == CodeStrategy.mb:
        for mb_cfg in generate_mb_cfg(args, cfg):
            yield mb_cfg
    elif code_strategy == CodeStrategy.ldpc:
        for ldpc_cfg in generate_ldpc_cfg(args, cfg):
            yield ldpc_cfg
    elif code_strategy == CodeStrategy.polar:
        for polar_cfg in generate_polar_cfg(args, cfg):
            yield polar_cfg
    else:
        raise "Unknown code strategy: " + str(code_strategy)


def generate_mb_cfg(args, cfg):
    if cfg.p_err == 0.0:
        success_rate_range = [1.0]
    elif args.mb_success_rate_range is not None:
        success_rate_range = args.mb_success_rate_range
    else:
        success_rate_range = [1.0 - 1.0 / args.N]
    for success_rate in success_rate_range:
        for block_size in args.mb_block_size_range:
            num_blocks = cfg.N // block_size
            actual_N = num_blocks * block_size
            for rounding_strategy in args.mb_rounding_strategy_list:
                if args.mb_goal_candidates_num is not None:
                    goal_candidates_num_range = [args.mb_goal_candidates_num]
                else:
                    goal_candidates_num_range = [3, 9, 27, 81, 243, 729, 2187, math.ceil(math.sqrt(actual_N))]
                for goal_candidates_num in goal_candidates_num_range:
                    for max_num_indices_to_encode in args.mb_max_num_indices_to_encode_range:
                        yield MbCfg(orig_cfg=cfg, success_rate=success_rate, block_length=block_size,
                                    num_blocks=num_blocks,
                                    goal_candidates_num=goal_candidates_num,
                                    indices_to_encode_strategy=IndicesToEncodeStrategy.most_candidate_blocks,
                                    rounding_strategy=rounding_strategy,
                                    pruning_strategy=args.mb_pruning_strategy,
                                    fixed_number_of_encodings=args.mb_fixed_number_of_encodings,
                                    max_num_indices_to_encode=max_num_indices_to_encode,
                                    radius_picking=args.mb_radius_picking,
                                    max_candidates_num=args.mb_max_candidates_num,
                                    encoding_sample_size=args.mb_encoding_sample_size)


def generate_ldpc_cfg(args, cfg):
    for sparsity in args.ldpc_sparsity_range:
        yield LdpcCfg(orig_cfg=cfg, sparsity=sparsity)


def generate_polar_cfg(args, cfg):
    for relative_gap_rate in args.polar_relative_gap_rate_list:
        if args.polar_scl_l_list is not None:
            scl_l_list = args.polar_scl_l_list
        else:
            scl_l_list = [3, 9, 27, 81, 243, 729, 2187]
        for scl_l in scl_l_list:
            yield PolarCfg(orig_cfg=cfg, constr_l=args.polar_constr_l, relative_gap_rate=relative_gap_rate, scl_l=scl_l)


def check_cfg_equality(cfg1):
    dict1 = cfg1.log_dict()
    specific_log_header_params = cfg.specific_log_header_params() + mb_cfg.specific_log_header_params() + ldpc_cfg.specific_log_header_params() + polar_cfg.specific_log_header_params()
    cfg_params = list(set(specific_log_header_params) & set(dict1.keys()))

    def check_cfg_equality_nested(cfg2):
        for key in cfg_params:
            if cfg2[key] != dict1[key]:
                return False
        return True

    return check_cfg_equality_nested
