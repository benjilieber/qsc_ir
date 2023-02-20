import csv
# import sys

import result
import run_module

def convert_txt_to_csv(input_txt_file_name):
    # args = sys.argv[1:]
    # input_txt_file_name = args[0]
    # output_csv_file_name = args[1]
    # output_agg_csv_file_name = args[1]
    # input_txt_file_name = "slurm-13884535.out"
    output_csv_file_name = "history.csv"
    output_agg_csv_file_name = "history_agg.csv"
    run_module.write_header(output_csv_file_name)
    run_module.write_header(output_agg_csv_file_name)
    input_txt_file = open(input_txt_file_name, 'r')
    input_txt_rows = input_txt_file.read().splitlines()
    with open(output_csv_file_name, 'a', newline='') as output_csv_file:
        with open(output_agg_csv_file_name, 'a', newline='') as output_agg_csv_file:
            assert(input_txt_rows[0] == str(result.get_header()))
            writer = csv.writer(output_csv_file)
            writer_agg = csv.writer(output_agg_csv_file)
            for input_txt_row in input_txt_rows[1:]:
                if input_txt_row[0] != "[":
                    continue
                cur_result = result.str_to_result(input_txt_row)
                if cur_result.sample_size == 1:
                    writer.writerow(result.str_to_result(input_txt_row).get_row())
                else:
                    writer_agg.writerow(result.str_to_result(input_txt_row).get_row())


# input_txt_file_name_list = ["slurm-13894838.out", "slurm-13894839.out", "slurm-13894842.out", "slurm-13894843.out", "slurm-13894846.out", "slurm-13894850.out", "slurm-13894866.out"]
# for input_txt_file_name in input_txt_file_name_list:
#     convert_txt_to_csv(input_txt_file_name)