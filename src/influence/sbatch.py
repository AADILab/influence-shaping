import os
import stat
from pathlib import Path
from typing import List
from influence.config import load_config

class FileInfo():
    def __init__(self, path, content):
        self.path = path
        self.content = content

def get_config_dirs(top_dir):
    config_dirs = []
    for dirpath, _, filenames in os.walk(os.path.expanduser(top_dir)):
        for filename in filenames:
            if filename == 'config.yaml':
                config_dir = Path(str(dirpath)+'/config.yaml')
                config_dirs.append(config_dir)
    return config_dirs

def generate_commands(config_dirs, seperate_trials):
    """Generate python commands to run configs in config_dirs"""
    commands = []
    for config_dir in config_dirs:
        command_start = 'python ~/influence-shaping/tools/run/config.py '
        command_end = ' --load_checkpoint'
        if seperate_trials:
            # Seperate trials means we generate a different command for running each trial
            config = load_config(config_dir)
            num_trials = config['experiment']['num_trials']
            for t in range(num_trials):
                command = command_start + '\'' + str(config_dir) + '\'' + ' -t ' + str(t) + command_end
                commands.append(command)
        else:
            # Running trials together means each config gets one command (rather than one command per trial)
            command = command_start + str(config_dir) + command_end
            commands.append(command)
    return commands

def generate_sh_command_dirs(out_root, commands):
    file_dirs = []

    spacer = 'trials'
    batch_dir = Path(str(out_root))/spacer

    for c in commands:
        if '-t' in c:
            split_c = c.split(' ')
            experiment_name = split_c[2].replace('/','.').replace('~.influence-shaping.results.', '').replace('.config.yaml','').replace("'",'') + '.t'+ str(split_c[4])
            # TODO: need to extract trial differently if it's in there
            file_name = experiment_name +'.sh'
            file_dir = batch_dir / file_name
            file_dirs.append(file_dir)
        else:
            experiment_name = c.split(' ')[-2].replace('/','.').replace('~.influence-shaping.results.', '').replace('.config.yaml','')
            file_name = experiment_name +'.sh'
            file_dir = batch_dir / file_name
            file_dirs.append(file_dir)

    return file_dirs

def write_command_sh_files(out_root, file_dirs, commands, file_str_start):
    if not os.path.exists( Path(os.path.expanduser(out_root)) /'trials' ):
        os.makedirs(Path(os.path.expanduser(out_root)) /'trials')
    for c, f in zip(commands, file_dirs):
        file_str = file_str_start + '\n' + c + '\n'
        with open(os.path.expanduser(f), 'w') as file:
            file.write(file_str)

def generate_sbatch_commands(bash_file_infos: List[FileInfo]):
    sbatch_commands = []
    for bash_file_info in bash_file_infos:
        sbatch_commands.append('sbatch ' + str(bash_file_info.path))
    return sbatch_commands

def write_sbatch_sh_file(out_root, sbatch_commands):
    batch_file_str = '\n'.join(sbatch_commands)+'\n'
    sbatch_dir = os.path.expanduser(out_root/'sbatch.sh')
    with open(sbatch_dir, 'w') as file:
        file.write(batch_file_str)
    os.chmod(sbatch_dir, os.stat(sbatch_dir).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def write_sbatch_executables_cli(top_dir: str, out_root: str, time: str, seperate_trials: bool):
    if len(os.listdir(top_dir)) == 0:
        print("Warning: Specified results directory is empty. Not writing out files.")
        return None

    # If out_root is None, then try to infer it from top_dir
    if out_root is None:
        # Infer out_dir by replacing 'results' with 'sbatch' in root_dir
        if 'results' not in top_dir:
            raise ValueError("No 'results' folder found in root_dir. sbatch_directory must be specified so sbatch files can be saved somewhere")
        out_root = top_dir.replace('results', 'sbatch')
    write_sbatch_executables(Path(top_dir), Path(out_root), time, seperate_trials)
    print(f"Successfully wrote top level executable to {out_root}/sbatch.sh")

def extract_path_after_results(path: Path):
    """Extract path after 'results' and convert to dot notation"""
    path_parts = path.parts
    try:
        results_index = path_parts.index('results')
        remaining_parts = path_parts[results_index + 1:]
        return '.'.join(remaining_parts)
    except ValueError:
        raise ValueError(f"'results' not found in path: {path}")

def generate_bash_files(config_dirs, out_root, time, seperate_trials):
    # Define the string you want to write
    common_sbatch_directives = [
        "#SBATCH -A kt-lab",
        "#SBATCH --partition=share,preempt",
        "#SBATCH -c 12",
        "#SBATCH --mem=16G",
        "#SBATCH --nodes=1",
        "#SBATCH --time="+time,
        "#SBATCH --requeue",
        "#SBATCH --nodelist=cn-v-[1-9],cn-t-1,cn-s-[1-5],cn-r-[1-4]"
    ]
    common_slurm_checks = [
        'echo "=== SLURM Directives ==="',
        'echo "  SLURM_JOB_ID: \$SLURM_JOB_ID"',
        'echo "  SLURM_JOB_NAME: \$SLURM_JOB_NAME"',
        'echo "  SLURM_CPUS_ON_NODE: \$SLURM_CPUS_ON_NODE"',
        'echo "  SLURM_MEM_PER_NODE: \$SLURM_MEM_PER_NODE"',
        'echo "  SLURM_JOB_NUM_NODES: \$SLURM_JOB_N,UM_NODES"'
        'echo "  SLURM_NODELIST: \$SLURM_NODELIST"',
        'echo "  SLURM_SUBMIT_DIR: \$SLURM_SUBMIT_DIR"',
        'echo "  SLURM_PARTITION: \$SLURM_JOB_PARTITION"'
    ]
    common_git_checks = [
        'cd ~/influence-shaping/',
        'echo "=== Running git commands for local ~/influence-shaping ==="',
        'echo "- GIT LOG -"',
        'git log --oneline -n 5 --no-color',
        'echo "- GIT STATUS -"',
        'git status',
        'echo "- GIT DIFF -"',
        'git --no-pager diff --no-color'
    ]
    common_python_checks = [
        'echo "===== Conda Environment Info ====="',
        'echo "- python --version -"',
        'python --version',
        'echo "- conda list -"',
        'conda list',
        'echo "- conda info -"',
        'conda info',
        'echo "- pip list -"',
        'pip list',
    ]

    # We should have one list of sbatch_directives for each config
    unique_sbatch_directives = []
    for config_dir in config_dirs:
        slurm_logs_dir = config_dir.parent / 'slurm_logs'
        sbatch_directives_for_config = [
            f'#SBATCH --output={slurm_logs_dir}/slurm-%j.out',
            f'#SBATCH --error={slurm_logs_dir}/slurm-%j.out'
        ]
        unique_sbatch_directives.append(sbatch_directives_for_config)

    bash_file_infos: List[FileInfo] = []
    for config_dir, sbatch_directives_for_config in zip(config_dirs, unique_sbatch_directives):
        start_file_lines = ['#!/bin/bash'] + \
            common_sbatch_directives + \
            sbatch_directives_for_config + \
            common_slurm_checks + \
            common_git_checks + \
            ['source /nfs/hpc/share/gonzaeve/miniforge/bin/activate influence'] + \
            common_python_checks
        if seperate_trials:
            config = load_config(config_dir)
            num_trials = config['experiment']['num_trials']
            for t in range(num_trials):
                # Figure out the path to save this bash file
                file_name = extract_path_after_results(config_dir)+f'.t{t}'+'.sh'
                path = out_root / 'bash' / file_name
                # Figure out the contents of this bash file
                command = ' '.join(['python', '~/influence-shaping/tools/run/config.py', str(config_dir), '--load_checkpoint', '-t', str(t)])
                content = '\n'.join(start_file_lines+[command])
                # Save it to file info list
                bash_file_infos.append(
                    FileInfo(
                        path=path,
                        content=content
                    )
                )
        else:
            # Figure out filename
            file_name = extract_path_after_results(config_dir)+'.sh'
            path = out_root / 'bash' / file_name
            # Figure out contents
            command = ' '.join(['python', '~/influence-shaping/tools/run/config.py', str(config_dir), '--load_checkpoint'])
            content = '\n'.join(start_file_lines+[command])
            # Save it to file info list
            bash_file_infos.append(
                FileInfo(
                    path=path,
                    content=content
                )
            )

    return bash_file_infos

def write_bash_files(out_root: Path, bash_file_infos: List[FileInfo]):
    (out_root/'bash').mkdir(parents=True, exist_ok=True)
    for bash_file_info in bash_file_infos:
        with open(os.path.expanduser(bash_file_info.path), 'w') as file:
            file.write(bash_file_info.content)

def write_sbatch_executables(top_dir: Path, out_root: Path, time: str, seperate_trials: bool):
    top_dir = top_dir.expanduser()
    out_root = out_root.expanduser()
    config_dirs = get_config_dirs(top_dir)
    bash_file_infos = generate_bash_files(config_dirs, out_root, time, seperate_trials)
    write_bash_files(out_root, bash_file_infos)

    sbatch_commands = generate_sbatch_commands(bash_file_infos)
    write_sbatch_sh_file(out_root, sbatch_commands)
