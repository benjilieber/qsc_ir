import os
import pandas as pd
import result

for old_file_name in os.listdir("results/polar_sample_10/"):
    if not old_file_name.endswith('csv'):
        continue
    print(old_file_name)
    old_df = pd.read_csv("results/polar_sample_10/" + old_file_name)
    new_df = pd.DataFrame(index=range(old_df.shape[0]), columns=result.get_header())
    for (columnName, columnData) in old_df.items():
        new_df[columnName] = old_df[columnName]
    new_df.cpu_time_rate = 0.0
    new_df.cpu_time_rate_success_only = 0.0
    new_file_name = "results/with_cpu_time/" + old_file_name
    new_df.to_csv(new_file_name, index=False)