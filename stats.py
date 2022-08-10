import csv
import itertools
import unittest
import numpy as np
from itertools import product
import scipy.special
import scipy.stats

import util
from key_generator import KeyGenerator
from linear_code_generator import LinearCodeGenerator
from protocol_configs import ProtocolConfigs


class StatsTest(unittest.TestCase):
    def test_solution_distribution(self):
        block_size = 22
        n = block_size
        m = 18
        p_err = 0
        cfg = ProtocolConfigs(base=3, block_length=block_size, num_blocks=1, p_err=p_err)
        encoding_matrix = LinearCodeGenerator(cfg).generate_encoding_matrix(m)
        keygen = KeyGenerator(p_err, n)
        a, b = keygen.generate_keys()

        deltas = product([1, 2], repeat=block_size)
        encoded_a_count = dict()
        for cur_delta in deltas:
            cur_candidate = np.mod(np.add(b, cur_delta), [3] * block_size).astype(int)
            encoded_a = np.matmul(cur_candidate, encoding_matrix) % 3
            encoded_a_count.setdefault(tuple(encoded_a), 0)
            encoded_a_count[tuple(encoded_a)] += 1

        cnt_distribution = [0 for _ in range(max(encoded_a_count.values()) + 1)]
        for cnt in encoded_a_count.values():
            cnt_distribution[cnt] += 1

        print(cnt_distribution)

    def test_avg_m(self):
        q = 3
        p = 0.02
        k_list = list(range(20))

        for k in k_list:
            avg_leak_space_size = (p ** (-p) * ((1-p)/(q-1))**(p-1)) ** k

            j = 0
            cur_space_size = 0
            while j <= k and cur_space_size < avg_leak_space_size:
                cur_space_size += scipy.special.binom(k, j) * (q-1) ** (k-j)
                j += 1
            block_success_prob = scipy.stats.binom.cdf(j, k, p)
            print("average radius = " + str(j) + ", block success prob = " + str(block_success_prob))

    def test_get_solution_list_size_dist(self):
        # file_name = "list_size_stats.csv"
        file_name = "list_size_stats_dev.csv"
        write_header(file_name, False)

        sample_size = 10
        q = 3
        for k in range(3, 16):
            for t in range(k + 1):
                print("k = " + str(k) + ", t = " + str(t))
                dist_list_no_zero, dist_list_with_zero = get_solution_list_size_dist_list(q, k, t, sample_size)
                # print("dist_list_no_zero = " + str(dist_list_no_zero))
                # print("dist_list_with_zero = " + str(dist_list_with_zero))
                # print("avg_values (no zero) = " + str(avg_values(dist_list_no_zero)))
                print("avg_values (with zero) = " + str(avg_values(dist_list_with_zero)))
                print("wished_values = " + str(wished_values(q, k, t)))
                for j in range(k + 1):
                    print("k = " + str(k) + ", t = " + str(t) + ", j = " + str(j))
                    dist_no_zero, dist_with_zero = get_solution_list_size_dist(q, k, j, t, sample_size)
                    # print("dist_no_zero = " + str(dist_no_zero))
                    # print("dist_with_zero = " + str(dist_with_zero))
                    # print("avg_value (no zero) = " + str(avg_value(dist_no_zero)))
                    print("avg_value (with zero) = " + str(avg_value(dist_with_zero)))
                    print("wished_value = " + str(wished_value(q, k, j, t)))
                    # write_result(file_name, q, k, j, t, avg_value(dist_no_zero), avg_value(dist_with_zero),
                    #              wished_value(q, k, j, t), sample_size, verbosity=False)

    def test_get_solution_list_size_dist_per_radius(self):
        # file_name = "list_size_stats_per_radius.csv"
        file_name = "list_size_stats_per_radius_dev.csv"
        write_header(file_name, True)

        sample_size = 10
        q = 3
        p = 0.01
        for k in range(3, 16):
            for t in range(k + 1):  # solution space complement dimension
                print("k = " + str(k) + ", t = " + str(t))
                dist_list_no_zero, dist_list_with_zero = get_solution_list_size_dist_list(q, k, t, sample_size)
                # print("dist_list_no_zero = " + str(dist_list_no_zero))
                # print("dist_list_with_zero = " + str(dist_list_with_zero))
                # print("avg_values (no zero) = " + str(avg_values(dist_list_no_zero)))
                print("avg_values (with zero) = " + str(avg_values(dist_list_with_zero)))
                print("wished_values = " + str(wished_values(q, k, t)))
                # write_result(file_name, q, k, t, avg_values(dist_list, k), wished_values(q, k, t), sample_size, verbosity=False)
                # print("exp_cum_value (with zero)" + str(exp_cum_values(dist_list_with_zero, p)))

    def test_get_solution_list_size_dist_per_radius2(self):
        # file_name = "list_size_stats_per_radius.csv"
        file_name = "list_size_stats_per_radius_dev.csv"
        write_header(file_name, True)

        sample_size = 10
        q = 3
        p_eq = 0.0
        for k in range(3, 16):
            for t in range(k + 1):  # solution space complement dimension
                print("k = " + str(k) + ", t = " + str(t))
                dist_list, cum_dist_list = get_solution_list_size_dist_list2(q, k, t, p_eq, sample_size)
                print("cum_avg (with zero) = " + str(avg_values(cum_dist_list)))
                print("wished_values = " + str(wished_cum_values(q, k, t)))

    def test_get_solution_list_size_dist_multi_block_all_solution_dimensions(self):
        # file_name = "list_size_stats_per_radius.csv"
        file_name = "list_size_stats_per_radius_dev.csv"
        write_header(file_name, True)

        single_block_sample_size = 100
        q = 3
        p_eq = 0.1
        success_rate = 0.9
        num_blocks = 100
        for k in range(3, 16):
            cfg = ProtocolConfigs(base=3, block_length=k, num_blocks=num_blocks, p_err=p_eq, success_rate=success_rate)
            for t in range(k + 1):  # solution space complement dimension
                print("k = " + str(k) + ", t = " + str(t))
                list_size_history = []
                single_block_list_dist, _ = get_solution_list_size_dist_list2(q, k, t, p_eq, single_block_sample_size)
                single_block_avg_list_sizes = avg_values(single_block_list_dist)
                # print("single_block_cum_avg (with zero) = " + str(single_block_avg_list_sizes))
                cur_prefix_avg_list_sizes = single_block_avg_list_sizes[:cfg.max_block_error[0]+1]
                for i in range(1, num_blocks):
                    cur_single_block_avg_list_sizes = single_block_avg_list_sizes[:cfg.max_block_error[i]+1]
                    cur_prefix_avg_list_sizes = [sum([avg_prefix_list_size * cur_single_block_avg_list_sizes[total_radius - (max(total_radius-(len(cur_single_block_avg_list_sizes)-1), 0) + j)] for j, avg_prefix_list_size in enumerate(cur_prefix_avg_list_sizes[max(total_radius-(len(cur_single_block_avg_list_sizes)-1), 0):total_radius+1])]) for total_radius in range(cfg.prefix_radii[i])]
                    # print("after " + str(i) + " blocks:" + str(sum(cur_prefix_avg_list_sizes)))
                    list_size_history.append(sum(cur_prefix_avg_list_sizes))
                print("history" + str(list_size_history))
                print("after: " + str(sum(cur_prefix_avg_list_sizes)))

    def test_get_solution_list_size_dist_multi_block_cfg_solution_dimension(self):
        # file_name = "list_size_stats_per_radius.csv"
        file_name = "list_size_stats_per_radius_dev.csv"
        write_header(file_name, True)

        single_block_sample_size = 100
        q = 3
        p_eq = 0.02
        success_rate = 0.9
        num_blocks = 10
        for k in range(3, 16):
            k = 10
            print("k = " + str(k))
            cfg = ProtocolConfigs(base=3, block_length=k, num_blocks=num_blocks, p_err=p_eq, success_rate=success_rate, fixed_number_of_encodings=True)
            single_block_avg_list_sizes = dict()
            for t in set(cfg.number_of_encodings_list):
                print("t = " + str(t))
                single_block_list_dist, _ = get_solution_list_size_dist_list2(q, k, t, p_eq, single_block_sample_size, max_radius=max(cfg.max_block_error))
                single_block_avg_list_sizes[t] = avg_values(single_block_list_dist)

            list_size_history = []
            cur_prefix_avg_list_sizes = single_block_avg_list_sizes[cfg.number_of_encodings_list[0]][:cfg.max_block_error[0]+1]
            for i in range(1, num_blocks):
                cur_single_block_avg_list_sizes = single_block_avg_list_sizes[cfg.number_of_encodings_list[i]][:cfg.max_block_error[i]+1]
                cur_prefix_avg_list_sizes = [sum([avg_prefix_list_size * cur_single_block_avg_list_sizes[total_radius - (max(total_radius-(len(cur_single_block_avg_list_sizes)-1), 0) + j)] for j, avg_prefix_list_size in enumerate(cur_prefix_avg_list_sizes[max(total_radius-(len(cur_single_block_avg_list_sizes)-1), 0):total_radius+1])]) for total_radius in range(cfg.prefix_radii[i])]
                # print("after " + str(i) + " blocks:" + str(sum(cur_prefix_avg_list_sizes)))
                list_size_history.append(sum(cur_prefix_avg_list_sizes))
            print("history: " + str(list_size_history))
            print("after: " + str(sum(cur_prefix_avg_list_sizes)))

def get_solution_list_size_dist(q, k, j, t, sample_size):
    list_size_counts = [0]
    for i in range(sample_size):
        image_count = dict()
        matrix = draw_matrix(q, k, t)

        for vec in get_nbhd(q, k, j):
            cur_img = np.matmul(vec, matrix) % q
            if image_count.get(tuple(cur_img)):
                image_count[tuple(cur_img)] += 1
            else:
                image_count[tuple(cur_img)] = 1
        if max(image_count.values()) >= len(list_size_counts):
            list_size_counts += [0 for _ in range(max(image_count.values()) + 1 - len(list_size_counts))]
        for cnt in image_count.values():
            list_size_counts[cnt] += 1
        list_size_counts[0] += q ** t - len(image_count)
    return np.array([0] + list_size_counts[1:]) / sum(list_size_counts[1:]), np.array(list_size_counts) / sum(list_size_counts)

def get_solution_list_size_dist_list(q, k, t, sample_size):
    list_size_counts_list = [[0] for _ in range(k + 1)]
    cum_list_size_count_list = [[0] for _ in range(k + 1)]
    for i in range(sample_size):
        image_count = [dict() for _ in range(k + 1)]
        cum_image_count = dict()
        matrix = draw_matrix(q, k, t)

        for j in range(k+1):
            for vec in get_nbhd_border(q, k, j):
                cur_img = np.matmul(vec, matrix) % q
                if image_count[j].get(tuple(cur_img)):
                    image_count[j][tuple(cur_img)] += 1
                else:
                    image_count[j][tuple(cur_img)] = 1
                if cum_image_count.get(tuple(cur_img)):
                    cum_image_count[tuple(cur_img)] += 1
                else:
                    cum_image_count[tuple(cur_img)] = 1
            if max(image_count[j].values()) >= len(list_size_counts_list[j]):
                list_size_counts_list[j] += [0 for _ in range(max(image_count[j].values()) + 1 - len(list_size_counts_list[j]))]
            if max(cum_image_count.values()) >= len(cum_list_size_count_list[j]):
                cum_list_size_count_list[j] += [0 for _ in range(max(cum_image_count.values()) + 1 - len(cum_list_size_count_list[j]))]
            for cnt in image_count[j].values():
                list_size_counts_list[j][cnt] += 1
            for cnt in cum_image_count.values():
                cum_list_size_count_list[j][cnt] += 1
            list_size_counts_list[j][0] += q ** t - len(image_count[j])
            cum_list_size_count_list[j][0] += q ** t - len(cum_image_count)
    print("cum_avg (with zero) = " + str(avg_values([np.array(list_size_counts) / sum(list_size_counts) for list_size_counts in cum_list_size_count_list])))

    return [np.array([0] + list_size_counts[1:]) / sum(list_size_counts[1:]) for list_size_counts in list_size_counts_list], [np.array(list_size_counts) / sum(list_size_counts) for list_size_counts in list_size_counts_list]

def get_solution_list_size_dist_list2(q, k, t, p_eq, sample_size, max_radius=None):
    per_radius_list_size_count_list = [[0] for _ in range(k + 1)]
    cum_list_size_count_list = [[0] for _ in range(k + 1)]
    b = np.zeros(k, dtype=int)

    for i in range(sample_size):
        a = KeyGenerator(p_eq, k, m=3).generate_complement_key(b)
        matrix = draw_matrix(q, k, t)
        a_img = np.matmul(a, matrix) % q
        a_candidates_num = 0

        if max_radius is None:
            max_radius = k
        for j in range(max_radius+1):
            a_candidates_num_delta = 0
            for vec in get_nbhd_border(q, k, j):
                if t == 0 or np.equal(np.matmul(vec, matrix) % q, a_img).all():
                    a_candidates_num_delta += 1
                    a_candidates_num += 1
            if a_candidates_num_delta >= len(per_radius_list_size_count_list[j]):
                per_radius_list_size_count_list[j] += [0 for _ in range(a_candidates_num_delta + 1 - len(per_radius_list_size_count_list[j]))]
            if a_candidates_num >= len(cum_list_size_count_list[j]):
                cum_list_size_count_list[j] += [0 for _ in range(a_candidates_num + 1 - len(cum_list_size_count_list[j]))]
            per_radius_list_size_count_list[j][a_candidates_num_delta] += 1
            cum_list_size_count_list[j][a_candidates_num] += 1
        # assert(cum_list_size_count_list[-1][q ** (k - t)] == i + 1)
    return [np.array(list_size_counts) / sum(list_size_counts) for list_size_counts in per_radius_list_size_count_list], [np.array(list_size_counts) / sum(list_size_counts) for list_size_counts in cum_list_size_count_list]

def draw_matrix(q, k, t):
    if t > k:
        raise RuntimeError("Can't generate matrix, max rank reached.")
    encoding_matrix = np.random.choice(range(q), (k, t))
    while util.calculate_rank(encoding_matrix, q) < t:
        encoding_matrix = np.random.choice(range(q), (k, t))
    return encoding_matrix

def get_nbhd(q, k, j):
    for n_err in range(j + 1):
        for candidate in get_nbhd_border(q, k, n_err):
            yield candidate

def get_nbhd_border(q, k, num_errors):
    deltas = product(list(range(1, q)), repeat=k - num_errors)
    for cur_delta in deltas:
        for err_indices in itertools.combinations_with_replacement(range(k - (num_errors - 1)), num_errors):
            yield np.insert(arr=cur_delta, obj=err_indices, values=0)

def avg_value(dist):
    return sum([i * prob for i, prob in enumerate(dist)])

def avg_values(dist_list):
    return [sum([i * prob for i, prob in enumerate(dist)]) for dist in dist_list]

def wished_value(q, k, j, t):
    return sum([scipy.special.binom(k, i) * (q-1) ** (k-i) for i in range(j+1)]) / q ** t

def wished_cum_values(q, k, t):
    wished_values_per_radius = wished_values(q, k, t)
    return [sum(wished_values_per_radius[:j+1]) for j in range(k + 1)]

def wished_values(q, k, t):
    return [scipy.special.binom(k, i) * (q-1) ** (k-i) / q ** t for i in range(k + 1)]

# def exp_cum_values(dist_list_with_zero, k, p):
#     weighted_expected_values = [scipy.special.binom(k, j) for j, dist in dist_list_with_zero]
#     return [ for j in enumerate(dist_list_with_zero)]

def write_header(file_name, per_radius):
    if per_radius:
        header = ["q", "block_size", "num_encodings", "average_list_sizes", "list_sizes_heuristic", "sample_size"]
    else:
        header = ["q", "block_size", "radius", "num_encodings", "average_list_size_no_zero", "average_list_size_with_zero", "list_size_heuristic", "sample_size"]
    try:
        with open(file_name, 'r') as f:
            for row in f:
                assert(row.rstrip('\n').split(",") == header)
                return
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")

def write_result(file_name, q, block_size, radius, num_encodings, average_list_size_no_zero, average_list_size_with_zero, expected_list_size, sample_size, verbosity=False):
    if verbosity:
        print("writing results")
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([q, block_size, radius, num_encodings, average_list_size_no_zero, average_list_size_with_zero, expected_list_size, sample_size])

if __name__ == '__main__':
    unittest.main()
