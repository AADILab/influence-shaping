import os
import stat
from pathlib import Path
import pprint
from influence.config import get_config_dirs, generate_commands, contractuser

def generate_sh_command_dirs(batch_dir_root, commands):
    file_dirs = []

    spacer = 'trials'
    batch_dir = Path(contractuser(str(batch_dir_root)))/spacer

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

def write_command_sh_files(batch_dir_root, file_dirs, commands, file_str_start):
    if not os.path.exists( Path(os.path.expanduser(batch_dir_root)) /'trials' ):
        os.makedirs(Path(os.path.expanduser(batch_dir_root)) /'trials')
    for c, f in zip(commands, file_dirs):
        file_str = file_str_start + '\n' + c + '\n'
        with open(os.path.expanduser(f), 'w') as file:
            file.write(file_str)

def generate_sbatch_commands(file_dirs):
    sbatch_commands = []
    for f in file_dirs:
        sbatch_commands.append('sbatch ' + str(f))
    return sbatch_commands

def write_sbatch_sh_file(batch_dir_root, sbatch_commands):
    batch_file_str = '\n'.join(sbatch_commands)+'\n'
    sbatch_dir = os.path.expanduser(batch_dir_root/'sbatch.sh')
    with open(sbatch_dir, 'w') as file:
        file.write(batch_file_str)
    os.chmod(sbatch_dir, os.stat(sbatch_dir).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def write_sbatch_executables_cli(top_dir: str, batch_dir_root: str, seperate_trials: bool):
    if len(os.listdir(top_dir)) == 0:
        print("Warning: Specified results directory is empty. Not writing out files.")
        return None

    # If batch_dir_root is None, then try to infer it from top_dir
    if batch_dir_root is None:
        # Infer out_dir by replacing 'results' with 'sbatch' in root_dir
        if 'results' not in top_dir:
            raise ValueError("No 'results' folder found in root_dir. sbatch_directory must be specified so sbatch files can be saved somewhere")
        batch_dir_root = top_dir.replace('results', 'sbatch')
    write_sbatch_executables(Path(top_dir), Path(batch_dir_root), seperate_trials)
    print(f"Successfully wrote top level executable to {batch_dir_root}/sbatch.sh")

def write_sbatch_executables(top_dir: Path, batch_dir_root: Path, seperate_trials: bool):
    # Define the string you want to write
    batch_file_start = [
        "#!/bin/bash",
        "#SBATCH -A kt-lab",
        "#SBATCH --partition=share,preempt",
        "#SBATCH -c 12",
        "#SBATCH --mem=16G",
        "#SBATCH --nodes=1",
        "#SBATCH --time=2-00:00:00",
        "#SBATCH --requeue",
        "#SBATCH --nodelist=cn-v-[1-9],cn-t-1,cn-s-[1-5],cn-r-[1-4]",
        "",
        "source ~/hpc-share/miniforge/bin/activate",
        "conda activate influence",
        ""
    ]

    file_str_start = '\n'.join(batch_file_start)

    # Leaving commented print statements for ease of debugging in the future
    config_dirs = get_config_dirs(top_dir)
    # pprint.pprint(config_dirs)
    commands = generate_commands(config_dirs, seperate_trials)
    # pprint.pprint(commands)
    file_dirs = generate_sh_command_dirs(batch_dir_root, commands)
    # pprint.pprint(file_dirs)
    sbatch_commands = generate_sbatch_commands(file_dirs)
    # pprint.pprint(sbatch_commands)
    write_command_sh_files(batch_dir_root, file_dirs, commands, file_str_start)
    write_sbatch_sh_file(batch_dir_root, sbatch_commands)
