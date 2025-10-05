#!/usr/bin/env bash
set -euo pipefail

# --- Usage and argument parsing ---
usage() {
    echo "Usage: $0 [DIRECTORY] [--dry-run]"
    echo
    echo "Finds pairs of CSV files:"
    echo "  champion_team_fitness.csv"
    echo "  resim_champion_team_fitness.csv"
    echo "and creates a merged file called:"
    echo "  fix_resim_champion_team_fitness.csv"
    echo
    echo "Options:"
    echo "  DIRECTORY   Directory to search (default: current directory)"
    echo "  --dry-run   Show what would be done, but don't write any files"
    exit 1
}

DRY_RUN=false
SEARCH_DIR="."

# Parse args
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -d "$arg" ]]; then
                SEARCH_DIR="$arg"
            else
                echo "Error: '$arg' is not a directory"
                usage
            fi
            ;;
    esac
done

echo "🔍 Searching in: $SEARCH_DIR"
$DRY_RUN && echo "🧪 Dry run mode: no files will be written."

# --- Main logic ---
find "$SEARCH_DIR" -type f -name "champion_team_fitness.csv" | while read -r oldfile; do
    dir=$(dirname "$oldfile")
    newfile="$dir/resim_champion_team_fitness.csv"
    fixedfile="$dir/fix_resim_champion_team_fitness.csv"

    if [[ -f "$newfile" ]]; then
        echo "Found pair in: $dir"
        echo "  - champion_team_fitness.csv"
        echo "  - resim_champion_team_fitness.csv"
        if $DRY_RUN; then
            echo "  → Would create: $fixedfile"
        else
            header=$(head -n 1 "$oldfile")
            {
                echo "$header"
                tail -n +2 "$newfile"
            } > "$fixedfile"
            echo "  ✅ Created: $fixedfile"
        fi
    else
        echo "⚠️  No resim_champion_team_fitness.csv found in $dir — skipping."
    fi
done

echo "✅ Done."
