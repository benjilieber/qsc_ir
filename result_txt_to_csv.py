import csv
import os
import re

import result
import run_module

def convert_txt_to_csv_single_file(input_txt_file_name):
    # args = sys.argv[1:]
    # input_txt_file_name = args[0]
    # output_csv_file_name = args[1]
    # output_agg_csv_file_name = args[1]
    # input_txt_file_name = "slurm-13884535.out"
    output_csv_file_name = "mb/results/history.csv"
    output_agg_csv_file_name = "mb/results/history_agg.csv"
    run_module.write_header(output_csv_file_name)
    run_module.write_header(output_agg_csv_file_name)
    input_txt_file = open(input_txt_file_name, 'r')
    input_txt_rows = input_txt_file.read().splitlines()
    with open(output_csv_file_name, 'a', newline='') as output_csv_file:
        with open(output_agg_csv_file_name, 'a', newline='') as output_agg_csv_file:
            assert(input_txt_rows[0] in [str(result.get_header()), str(result.get_old_header())])
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

def convert_txt_to_csv():
    for filename in os.listdir():
        if re.match(r'*.out', filename):
            convert_txt_to_csv_single_file(filename)