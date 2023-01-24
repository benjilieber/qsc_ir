import csv
import os
import sys
import pickle

import result

sys.path.append(os.getcwd())
import numpy as np
import time
import multiprocessing

from timeit import default_timer as timer
from protocol_configs import IndicesToEncodeStrategy, CodeGenerationStrategy
from protocol_configs import ProtocolConfigs
from key_generator import KeyGenerator
from multi_block_protocol import MultiBlockProtocol
from result import Result
import math


def write_header(file_name):
    try:
        with open(file_name, 'r') as f:
            for row in f:
                assert(row.rstrip('\n').split(",") == result.get_header())
                return
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result.get_header())
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")

def write_results(result_list, is_slurm=False, verbosity=False):
    if verbosity:
        print("writing results")
    if is_slurm:
        for single_result in result_list:
            print(single_result.get_row())
        print(Result(result_list[0].cfg, result_list=result_list).get_row())
        return
    raw_results_file_name = result_list[0].cfg.raw_results_file_path
    with open(raw_results_file_name, 'a', newline='') as f1:
        writer = csv.writer(f1)
        for single_result in result_list:
            writer.writerow(single_result.get_row())

    agg_results_file_name = result_list[0].cfg.agg_results_file_path
    with open(agg_results_file_name, 'a', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(Result(result_list[0].cfg, result_list=result_list).get_row())

def single_run(cfg):
    # print(f"I'm process {os.getpid()}")
    np.random.seed([os.getppid(), int(str(time.time() % 1)[2:10])])
    key_generator = KeyGenerator(p_err=cfg.p_err, key_length=cfg.key_length, base=cfg.base)
    a, b = key_generator.generate_keys()
    protocol = MultiBlockProtocol(cfg, a, b)
    return protocol.run()

def multi_run_series(cfg, sample_size, is_slurm, verbosity=False):
    result_list = []
    for single_sample_run in range(sample_size):
        result = single_run(cfg)
        if verbosity:
            print(result)
        result_list.append(result)
    if verbosity:
        print(Result(cfg, result_list=result_list))
    write_results(result_list, is_slurm=is_slurm, verbosity=verbosity)


def multi_run_parallel(cfg, sample_size, is_slurm, verbosity=False):
    values = [cfg] * sample_size
    with multiprocessing.Pool(sample_size) as pool:
    # with multiprocessing.Pool() as pool:
        write_results(pool.map(single_run, values), is_slurm=is_slurm, verbosity=verbosity)
        # print("starting to run")\
        # return pool.map_async(single_run, values, callback=write_results)


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

    for code_generation_strategy in args.code_generation_strategy_list:
        assert (code_generation_strategy == CodeGenerationStrategy.LINEAR_CODE)
        for q in args.q_list:
            for key_size in args.key_size_list:
                for block_size in args.block_size_range:
                    num_blocks = key_size // block_size
                    actual_key_size = num_blocks * block_size
                    for p_err in args.p_err_range:
                        success_rate_range = args.success_rate_range or [1.0 - 1.0/actual_key_size]
                        for success_rate in success_rate_range:
                            for rounding_strategy in args.rounding_strategy_list:
                                # goal_candidates_num_range = [args.goal_candidates_num] or [2 ** i for i in range(math.ceil(math.log(actual_key_size, 2))+2)]
                                if args.fixed_number_of_encodings:
                                    goal_candidates_num_range = [None]
                                else:
                                    goal_candidates_num_range = [args.goal_candidates_num] if (args.goal_candidates_num is not None) else [math.ceil(math.sqrt(key_size))]
                                for goal_candidates_num in goal_candidates_num_range:
                                    for max_num_indices_to_encode in args.max_num_indices_to_encode_range:
                                        for sparsity in args.sparsity_range or [0]:
                                            try:
                                                cfg = ProtocolConfigs(base=q, block_length=block_size, num_blocks=num_blocks,
                                                                  p_err=p_err,
                                                                  success_rate=success_rate,
                                                                  goal_candidates_num=goal_candidates_num,
                                                                  indices_to_encode_strategy=IndicesToEncodeStrategy.MOST_CANDIDATE_BLOCKS,
                                                                  rounding_strategy=rounding_strategy,
                                                                  code_generation_strategy=code_generation_strategy,
                                                                  pruning_strategy=args.pruning_strategy,
                                                                  sparsity=sparsity,
                                                                  fixed_number_of_encodings=args.fixed_number_of_encodings,
                                                                  max_num_indices_to_encode=max_num_indices_to_encode,
                                                                  radius_picking=args.radius_picking,
                                                                  max_candidates_num=args.max_candidates_num,
                                                                  encoding_sample_size=args.encoding_sample_size,
                                                                  raw_results_file_path=args.raw_results_file_path,
                                                                  agg_results_file_path=args.agg_results_file_path,
                                                                  verbosity=args.verbosity)
                                                if args.verbosity:
                                                    print(result.Result(cfg))
                                                if args.run_mode == 'parallel':
                                                    # r_list.append(multi_run_parallel(cfg, args.sample_size))
                                                    multi_run_parallel(cfg, args.sample_size, is_slurm=args.is_slurm, verbosity=args.verbosity)
                                                else:
                                                    multi_run_series(cfg, args.sample_size, is_slurm=args.is_slurm, verbosity=args.verbosity)
                                            except TimeoutError:
                                                continue

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