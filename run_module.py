import csv
import math
import multiprocessing
import os
import sys
import time
from itertools import product
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import cfg
import result
from cfg import Cfg, CodeStrategy
from key_generator import KeyGenerator
from ldpc import ldpc_cfg
from ldpc.ldpc_cfg import LdpcCfg
from ldpc.ldpc_protocol import LdpcProtocol
from mb import mb_cfg
from mb.mb_cfg import IndicesToEncodeStrategy
from mb.mb_cfg import MbCfg
from mb.mb_protocol import MbProtocol
from polar import polar_cfg
from polar.polar_cfg import PolarCfg
from polar.polar_protocol import PolarProtocol
from result import Result

sys.path.append(os.getcwd())


def write_header(file_name):
    try:
        with open(file_name, 'r') as f:
            csv_reader = csv.DictReader(f)
            list_of_column_names = set(csv_reader.fieldnames)
            assert (list_of_column_names == set(result.get_header()))
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=result.get_header())
            csv_writer.writeheader()
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")


def write_results(result_list, verbosity=False):
    if verbosity:
        print("writing results")
    raw_results_file_name = result_list[0].cfg.raw_results_file_path
    with open(raw_results_file_name, 'a', newline='') as f1:
        writer = csv.DictWriter(f1, fieldnames=result.get_header())
        for single_result in result_list:
            writer.writerow(single_result.get_dict())

    agg_results_file_name = result_list[0].cfg.agg_results_file_path
    with open(agg_results_file_name, 'a', newline='') as f2:
        writer = csv.DictWriter(f2, fieldnames=result.get_header())
        writer.writerow(Result(result_list[0].cfg, result_list=result_list).get_dict())


def write_results_parallel_helper(result_tuple_list, verbosity=False):
    for i in range(len(result_tuple_list[0])):
        result_i_list = [result_tuple[i] for result_tuple in result_tuple_list]
        write_results(result_i_list, verbosity=verbosity)

def single_run(cfg):
    # print(f"Started process {os.getpid()}", flush=True)
    np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    key_generator = KeyGenerator(p_err=cfg.p_err, N=cfg.N, base=cfg.q)
    a, b = key_generator.generate_keys()
    if cfg.code_strategy == CodeStrategy.mb:
        protocol = MbProtocol(cfg, a, b)
    elif cfg.code_strategy == CodeStrategy.ldpc:
        protocol = LdpcProtocol(cfg, a, b)
    elif cfg.code_strategy == CodeStrategy.polar:
        protocol = PolarProtocol(cfg, a, b)
    else:
        raise "Unknown code strategy: " + str(cfg.code_strategy)
    result_tuple = protocol.run()
    # print(f"Ended process {os.getpid()}", flush=True)
    return result_tuple


def multi_run_series(cfg, sample_size, verbosity=False):
    result_tuple_list = []
    for single_sample_run in range(sample_size):
        result_tuple = single_run(cfg)
        result_tuple_list.append(result_tuple)
    for i in range(len(result_tuple_list[0])):
        result_i_list = [result_tuple[i] for result_tuple in result_tuple_list]
        write_results(result_i_list, verbosity=verbosity)


def multi_run_parallel(cfg, sample_size, verbosity=False):
    values = [cfg] * sample_size
    # partial_write_results = partial(write_results, verbosity=verbosity)
    with multiprocessing.Pool(sample_size) as pool:
        # with multiprocessing.Pool() as pool:
        write_results_parallel_helper(pool.map(single_run, values), verbosity=verbosity)
        if verbosity:
            print("starting to run")
        # return pool.map_async(single_run, values, callback=partial_write_results)


def multi_run(args):
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
        if os.path.isfile(args.previous_run_files[0]):
            assert (len(args.previous_run_files) <= 1)
            previous_cfg_df = pd.read_csv(args.previous_run_files[0])
            if previous_cfg_df.empty:
                args.previous_run_files = None
        else:
            args.previous_run_files = None

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
                            print(run_cfg)
                        if args.run_mode == 'parallel':
                            # r_list.append(multi_run_parallel(cfg, args.sample_size))
                            multi_run_parallel(run_cfg, args.sample_size,
                                               verbosity=args.verbosity)
                        else:
                            multi_run_series(run_cfg, args.sample_size,
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
            if ldpc_cfg.syndrome_length < ldpc_cfg.N:
                yield ldpc_cfg
    elif code_strategy == CodeStrategy.polar:
        for polar_cfg in generate_polar_cfg(args, cfg):
            if polar_cfg.num_info_indices > 0:
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
                    goal_candidates_num_range = [3, 9, 27, 81, 243, 729]  # , 2187, math.ceil(math.sqrt(actual_N))
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
    if args.ldpc_key_rate_list is None:
        args.ldpc_key_rate_list = [None]
    if args.ldpc_syndrome_length_list is None:
        args.ldpc_syndrome_length_list = [None]
    if args.ldpc_success_rate_list is None:
        args.ldpc_success_rate_list = [None]
    if args.ldpc_relative_gap_rate_list is None:
        args.ldpc_relative_gap_rate_list = [None]

    for key_rate, syndrome_length, success_rate, relative_gap_rate in product(args.ldpc_key_rate_list,
                                                                              args.ldpc_syndrome_length_list,
                                                                              args.ldpc_success_rate_list,
                                                                              args.ldpc_relative_gap_rate_list):
        for sparsity in args.ldpc_sparsity_range:
            for decoder in args.ldpc_decoder_list:
                for max_num_rounds in args.ldpc_max_num_rounds_list:
                    for L in args.ldpc_L_list:
                        for use_forking in args.ldpc_use_forking_list:
                            for use_hints in args.ldpc_use_hints_list:
                                yield LdpcCfg(orig_cfg=cfg,
                                              key_rate=key_rate,
                                              syndrome_length=syndrome_length,
                                              success_rate=success_rate,
                                              relative_gap_rate=relative_gap_rate,
                                              sparsity=sparsity,
                                              decoder=decoder,
                                              max_num_rounds=max_num_rounds,
                                              L=L,
                                              use_forking=use_forking,
                                              use_hints=use_hints)


def generate_polar_cfg(args, cfg):
    if args.polar_key_rate_list is None:
        args.polar_key_rate_list = [None]
    if args.polar_num_info_indices_list is None:
        args.polar_num_info_indices_list = [None]
    if args.polar_success_rate_list is None:
        args.polar_success_rate_list = [None]
    if args.polar_relative_gap_rate_list is None:
        args.polar_relative_gap_rate_list = [None]

    for key_rate, num_info_indices, success_rate, relative_gap_rate in product(args.polar_key_rate_list,
                                                                               args.polar_num_info_indices_list,
                                                                               args.polar_success_rate_list,
                                                                               args.polar_relative_gap_rate_list):
        if args.polar_scl_l_list is not None:
            scl_l_list = args.polar_scl_l_list
        else:
            scl_l_list = [3, 9, 27, 81, 243, 729, 2187]
        for scl_l in scl_l_list:
            if scl_l > 1:
                check_length_list = [0, int(math.ceil(math.log(scl_l, cfg.q)))]
            else:
                check_length_list = [0]
            for check_length in check_length_list:
                yield PolarCfg(orig_cfg=cfg,
                               constr_l=args.polar_constr_l,
                               key_rate=key_rate,
                               num_info_indices=num_info_indices,
                               success_rate=success_rate,
                               relative_gap_rate=relative_gap_rate,
                               scl_l=scl_l,
                               check_length=check_length)


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
