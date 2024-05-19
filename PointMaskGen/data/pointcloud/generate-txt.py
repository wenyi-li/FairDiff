import os
import argparse


def get_subfolder_names(folder_path):
    if not os.path.isdir(folder_path):
        print("Provided path is not a directory.")
        return []
    subfolder_names = [name for name in os.listdir(folder_path)
                       if os.path.isdir(os.path.join(folder_path, name))]
    return subfolder_names

def generate_file_list_txt(base_directory, sub_directory, output_txt):
    directory = os.path.join(base_directory, sub_directory)
    with open(output_txt, 'w') as file:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                file.write(filename + '\n')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--categories', type=str, default='gender',
                    choices = ['ethnicity', 'gender', 'language', 'maritalstatus',
                    'race'])
args = parser.parse_args()
base_directory = os.getcwd()
folder = os.path.join(base_directory, args.categories)
sub_directories = get_subfolder_names(folder)

for sub_directory in sub_directories:
    output_txt = os.path.join(folder, f'{sub_directory}.txt')
    generate_file_list_txt(folder, sub_directory, output_txt)
