"""Compress the input directory using tar and gzip"""

import argparse
import tarfile
import os
from tqdm import tqdm
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress directory into tar.gz file',
        epilog=''
    )
    parser.add_argument('directory', help='Directory to compress')
    args = parser.parse_args()

    start_dir = Path(os.path.expanduser(args.directory))
    compressed_dir = Path(os.path.expanduser(args.directory)+'.tar.gz')

    paths_to_compress = []
    arcnames = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            absolute_path = Path(root)/file
            relative_path = os.path.relpath(absolute_path, start_dir)
            paths_to_compress.append(absolute_path)
            arcnames.append(relative_path)

    with tarfile.open(compressed_dir, 'w:gz') as tar:
        for arcname, path in tqdm(list(zip(arcnames, paths_to_compress))):
            tar.add(path, arcname=arcname)
