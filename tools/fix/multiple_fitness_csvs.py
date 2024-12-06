import argparse
import os
from pathlib import Path

def fix_csv(csv_dir: Path, out_dir: Path):

    # Figure out where the comma is missing and insert it
    #  You know a comma is missing if there are two '.' characters in one floating point number
    #  So...
    #  Split the line into a list seperated by commas
    #  Go into each element. Check if it is correct or needs fixing
    #  Add either the element itself or the fixed element to a new list
    #  This new list will be what we write out to the new csv

    # with open(csv_dir, newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         print(row)

    with open(csv_dir, 'r') as file:
        # List of strings. Each string is a line in the file
        lines = file.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        # First line is the header. Leave as is
        if i == 0:
            new_lines.append(line)
        # Otherwise get to work
        else:
            new_line_list = []
            list_line = line.split(',')
            for num in list_line:
                # If this number has an appropriate amount of 
                # '.' characters, leave it.
                if num.count('.') <= 1:
                    new_line_list.append(num)
                # Otherwise, there are too many '.' characters. Fix it.
                else:
                    dot_inds = [i for i, c in enumerate(num) if c == '.']
                    second_ind = dot_inds[1]
                    new_nums = [num[:second_ind-1], num[second_ind-1:]]
                    new_line_list+=new_nums
            new_line = ','.join(new_line_list)
            new_lines.append(new_line)
        
    # Write it all out
    with open(out_dir, 'w') as file:
        for new_line in new_lines:
            file.write(new_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='fitness_csv.py',
        description='Fix all the csv files in a directory tree so we can look at fitnesses in different rollouts. WARNING: This will overwrite existing csvs',
        epilog=''
    )
    parser.add_argument(
        'root_dir',
        help='root directory of csvs that needs to be fixed',
        type=str
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    # Aggregate all the directories of csvs that need fixing
    csv_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'fitness.csv' in filenames:
            csv_dirs.add(Path(dirpath) / 'fitness.csv')
    
    # Fix em!
    for csv_dir in csv_dirs:
        fix_csv(csv_dir=csv_dir, out_dir=csv_dir)
