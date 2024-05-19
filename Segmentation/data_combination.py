import pandas as pd
import os
import random

##### Equal Scale Data Combination #######
# Real: White 1521, Black 295, Asian 184
# Syn:  White 1145, Black 2372, Asian 2483
# All:  White 2666, Black 2667, Asian 2667
##########################################

# Real data
can_be_replaced = True

df = pd.read_csv('path_to_Harvard_FairSeg/data_summary.csv')

White_df = df[(df['race'] == 'White') & (df['train'] == 1)]
White_filenames = White_df['filename']
White_sample = White_filenames.sample(n=1521, random_state=42, replace=can_be_replaced) 

Black_df = df[(df['race'] == 'Black') & (df['train'] == 1)]
Black_filenames = Black_df['filename']
Black_sample = Black_filenames.sample(n=295, random_state=42, replace=can_be_replaced) 

Asian_df = df[(df['race'] == 'Asian') & (df['train'] == 1)]
Asian_filenames = Asian_df['filename']
Asian_sample = Asian_filenames.sample(n=184, random_state=42, replace=can_be_replaced) 

merged_samples = pd.concat([White_sample, Black_sample, Asian_sample], ignore_index=True)

with open('combined.txt', 'w') as file:
    for filename in merged_samples:
        file.write(f"data/{filename}\n")


# Syn data
White_dir = 'path_to_white_data'
Black_dir = 'path_to_black_data'
Asian_dir = 'path_to_asian_data'

White_data = os.listdir(White_dir)
Black_data = os.listdir(Black_dir)
Asian_data = os.listdir(Asian_dir)

White_sample = random.sample(White_data, 1145)
Black_sample = random.sample(Black_data, 2372)
Asian_sample = random.sample(Asian_data, 2483)

with open('combined.txt', 'a') as file:
    for s in White_sample:
        file.write(f"path/race/White/data/{s}\n")

    for s in Black_sample:
        file.write(f"path/race/Black/data/{s}\n")

    for s in Asian_sample:
        file.write(f"path/race/Asian/data/{s}\n")