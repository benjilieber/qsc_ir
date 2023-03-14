import math
import unittest

import result
from mb.mb_cfg import MbCfg
from result import Result, str_to_result, result_str_to_cfg_str


class ResultTest(unittest.TestCase):
    def test_single_result(self):
        n = 1000
        block_length = 8
        num_blocks = math.ceil(n / block_length)
        p_err = 0.05
        success_rate = 0.99
        max_candidates_num = 50
        max_num_indices_to_encode = 3
        cfg = MbCfg(q=3, block_length=block_length, num_blocks=num_blocks, p_err=p_err,
                    success_rate=success_rate, max_candidates_num=max_candidates_num,
                    max_num_indices_to_encode=max_num_indices_to_encode)
        is_success = True
        key_rate = 0.6
        communication_rate = 42
        time = 56
        r = Result(cfg, is_success=is_success, key_rate=key_rate, leak_rate=communication_rate,
                   matrix_size_rate=communication_rate, bob_communication_rate=communication_rate,
                   total_communication_rate=communication_rate, time_rate=time)

        expected_theoretic_key_rate = math.log(3, 2) + p_err * math.log(p_err, 2) + (1 - p_err) * math.log(
            (1 - p_err) / 2, 2)
        expected_cfg_row = [n, block_length, num_blocks, p_err, success_rate, max_candidates_num,
                            max_num_indices_to_encode, 'mb', expected_theoretic_key_rate]
        expected_output_row = [is_success, key_rate, communication_rate, communication_rate, communication_rate,
                               communication_rate, time]
        expected_row = expected_cfg_row + expected_output_row

        self.assertEqual(expected_cfg_row, r.get_cfg_row())
        self.assertEqual(expected_output_row, r.get_output_row())
        self.assertEqual(expected_row, r.get_row())

    def test_aggregated_results(self):
        n = 1000
        block_length = 8
        num_blocks = math.ceil(n / block_length)
        p_err = 0.05
        success_rate = 0.99
        max_candidates_num = 50
        max_num_indices_to_encode = 3
        cfg = MbCfg(q=3, block_length=block_length, num_blocks=num_blocks, p_err=p_err,
                    success_rate=success_rate, max_candidates_num=max_candidates_num,
                    max_num_indices_to_encode=max_num_indices_to_encode)

        r_list = []
        for i in range(10):
            is_success = (i >= 5)
            key_rate = 0.42 + 0.02 * i
            communication_rate = 41 + 1 * i
            time = 53 + 3 * i
            r_list.append(Result(cfg, is_success=is_success, key_rate=key_rate, leak_rate=communication_rate,
                                 matrix_size_rate=communication_rate, bob_communication_rate=communication_rate,
                                 total_communication_rate=communication_rate, time_rate=time))

        expected_theoretic_key_rate = math.log(3, 2) + p_err * math.log(p_err, 2) + (1 - p_err) * math.log(
            (1 - p_err) / 2, 2)
        expected_cfg_row = [n, block_length, num_blocks, p_err, success_rate, max_candidates_num,
                            max_num_indices_to_encode, 'mb', expected_theoretic_key_rate]
        expected_output_row = [10, 0.5, 0.51, 45.5, 45.5, 45.5, 45.5, 66.5]
        expected_row = expected_cfg_row + expected_output_row

        self.assertEqual(expected_row, Result(cfg, result_list=r_list).get_row())

    def test_single_result_from_string(self):
        N = 1000
        block_length = 8
        num_blocks = math.ceil(N / block_length)
        p_err = 0.05
        success_rate = 0.99
        max_candidates_num = 50
        max_num_indices_to_encode = 3
        cfg = MbCfg(q=3, block_length=block_length, num_blocks=num_blocks, p_err=p_err,
                    success_rate=success_rate, max_candidates_num=max_candidates_num,
                    max_num_indices_to_encode=max_num_indices_to_encode)
        is_success = True
        key_rate = 0.6
        communication_rate = 42
        time = 56
        r = Result(cfg, is_success=is_success, key_rate=key_rate, leak_rate=communication_rate,
                   matrix_size_rate=communication_rate, bob_communication_rate=communication_rate,
                   total_communication_rate=communication_rate, time_rate=time)

        r_clone = str_to_result(str(r.get_row()))

        self.assertEqual(r.get_row(), r_clone.get_row())

    def test_cfg_string_from_row_string(self):
        N = 1000
        block_length = 8
        num_blocks = math.ceil(N / block_length)
        p_err = 0.05
        success_rate = 0.99
        max_candidates_num = 50
        max_num_indices_to_encode = 3
        cfg = MbCfg(q=3, block_length=block_length, num_blocks=num_blocks, p_err=p_err,
                    success_rate=success_rate, max_candidates_num=max_candidates_num,
                    max_num_indices_to_encode=max_num_indices_to_encode)
        is_success = True
        key_rate = 0.6
        communication_rate = 42
        time = 56
        r = Result(cfg, is_success=is_success, key_rate=key_rate, leak_rate=communication_rate,
                   matrix_size_rate=communication_rate, bob_communication_rate=communication_rate,
                   total_communication_rate=communication_rate, time_rate=time)

        cfg_string = result_str_to_cfg_str(str(r.get_row()))

        self.assertEqual(str(r.get_cfg_row()), cfg_string)

    def test_convert_output_files_to_cfg_string_list(self):
        print(result.convert_output_files_to_cfg_string_list(["fake_results.csv"], "csv"))


if __name__ == '__main__':
    unittest.main()
