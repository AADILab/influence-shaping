#!/usr/bin/env bash

# Top-level directory (can be passed as an argument, defaults to current dir)
TOP_DIR="${1:-.}"

# Find all fitness.csv files
find "$TOP_DIR" -type f -name "fitness.csv" | while read -r file; do
    # Count lines in the file
    line_count=$(wc -l < "$file")
    
    # Check if the line count is exactly 1003
    if [ "$line_count" -ne 1002 ]; then
        echo "Incorrect length: $file (lines: $line_count)"
    fi
done
