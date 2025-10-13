import yaml
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Any

def load_config(config_dir):
    with open(os.path.expanduser(config_dir), 'r') as file:
        return yaml.safe_load(file)

def write_config(config, config_dir):
    folder_dir = Path(os.path.expanduser(config_dir)).parent
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir, exist_ok = True)
    with open(config_dir, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def config_command_in_dict(dict_: dict):
    for key in dict_:
        if key[0] == '~':
            return True
    return False

def get_value(dict_: dict, keys: list[str]):
    if len(keys) == 1:
        return dict_[keys[0]]
    else:
        return get_value(dict_, keys=keys[1:])

def get_replace_unique_keys_items(dict_: dict):
    # NOTE: This only works when each nested dictionary has only one key
    if isinstance(dict_, dict):
        for key in dict_:
            keys, items = get_replace_unique_keys_items(dict_[key])
            return [key]+keys, items
    else:
        return [], dict_

def set_value(dict_: dict, keys: list[str], value: Any):
    if len(keys) == 1:
        dict_[keys[0]] = value
    else:
        if keys[0] not in dict_:
            dict_[keys[0]] = {}
        set_value(dict_[keys[0]], keys[1:], value)

def expand_replace_unique(dict_: dict):
    keys, items = get_replace_unique_keys_items(dict_)
    item_dicts = []
    for item in items:
        item_dict = {}
        set_value(item_dict, keys[1:], item)
        item_dicts.append(item_dict)
    # del dict_['~replace_unique']
    return item_dicts

def merge_dicts(dict1, dict2):
    """Merge dict2 into dict1 recursively (ie: any nested dictionaries). For any conflicts, dict2 overwrites dict1"""
    keys = [k for k in dict2]
    for key in dict2:
        # If dict 2 contains a command, then execute it
        # if isinstance(dict2[key], dict) and '~fill_duplicate' in dict2[key]:
        if isinstance(dict2[key], dict) and config_command_in_dict(dict2[key]):
            if '~replace_duplicate' in dict2[key] and isinstance(dict1[key], list):
                for i in range(len(dict1[key])):
                    merge_dicts(dict1[key][i], deepcopy(dict2[key]['~replace_duplicate']))
                del dict2[key]['~replace_duplicate']
            if '~replace_unique' in dict2[key] and isinstance(dict1[key], list):
                dict2[key] = expand_replace_unique(dict2[key])
                for i in range(len(dict1[key])):
                    merge_dicts(dict1[key][i], deepcopy(dict2[key][i]))
                # dict1[key] = dict2[key]
                # merge_dicts(dict1, dict2)
        # If both values are dictionaries, merge them recursively
        elif key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dicts(dict1[key], dict2[key])
        # Otherwise, update/overwrite dict1 using the value from dict2
        else:
            dict1[key] = dict2[key]

def expand_keys(dict_):
    """Expand the keys of a given dict from ['key1:key2:key3'] to ['key1']['key2']['key3'] in-place"""
    keys = [k for k in dict_]
    for key in keys:
        if ":" in key:
            keys = key.split(":")
            first_key = keys[0]
            new_dict = expand_keys({":".join(keys[1:]) : dict_[key]})
            if first_key in dict_:
                merge_dicts(dict_[first_key], new_dict)
            else:
                dict_[first_key] = expand_keys({":".join(keys[1:]) : dict_[key]})
            del dict_[key]
        elif isinstance(dict_[key], dict):
            expand_keys(dict_[key])
        elif isinstance(dict_[key], list):
            # Note: This does not check recursively for lists.
            # Keys within lists of lists of dictionaries will not get expanded
            for element in dict_[key]:
                if isinstance(element, dict):
                    expand_keys(element)
    return dict_

def merge_dicts_list(list_of_dicts):
    if len(list_of_dicts) == 1:
        return list_of_dicts[0]
    else:
        # return merge_base(list_of_dicts[0], merge_dicts_list(list_of_dicts[1:]))
        return merge_base(merge_dicts_list(list_of_dicts[:-1]), list_of_dicts[-1])

def merge_base(dict1, dict2):
    new_dict = deepcopy(dict1)
    merge_dicts(new_dict, deepcopy(dict2))
    return new_dict

def consolidate_parameters(parameter_dicts, addtl_list=[]):
    consolidated_dict = {}
    for key in parameter_dicts[0]:
        if len(parameter_dicts) == 1:
            consolidated_dict[key] = merge_dicts_list(addtl_list+[parameter_dicts[0][key]])
        else:
            consolidated_dict[key] = consolidate_parameters(parameter_dicts[1:], addtl_list+[parameter_dicts[0][key]])
    return consolidated_dict

# Now turn this into a list of directories with the corresponding config we are going to save there
def create_directory_dict(consolidated_dict, path_len, path_list=[], directory_dict = {}):
    """Create a dictionary where the key is the path to the config,
    and the value is the dictionary of unique parameters that will be saved in that config
    """
    for key in consolidated_dict:
        new_path_list = path_list + [key]
        if len(path_list) >= path_len:
            directory_dict[os.path.join(*new_path_list)] = consolidated_dict[key]
        else:
            create_directory_dict(consolidated_dict[key], path_len, new_path_list, directory_dict)

    return directory_dict

def expand_directory_dict(directory_dict, base_config):
    """Merge individual parameter configs with the base config"""
    # NOTE: Unique parameters overwrite base parameters
    for dir_str in directory_dict:
        directory_dict[dir_str] = merge_base(base_config, directory_dict[dir_str])

def write_directory_dict(directory_dict, top_dir):
    for dir in directory_dict:
        config_dir = Path(os.path.expanduser(top_dir))/dir/'config.yaml'
        write_config(directory_dict[dir], config_dir)

def write_config_tree_cli(sweep_config_dir, top_write_dir: Optional[str] = None):
    # If top_write_dir is None, then try to infer it from sweep_config_dir
    if top_write_dir is None:
        # Infer out_dir by replacing 'tree_gen' with 'results' in root_dir
        if 'tree_gen' not in sweep_config_dir:
            raise ValueError("No 'tree_gen' folder found in root_dir. top_write_directory must be specified so configs can be saved somewhere")
        top_write_dir = sweep_config_dir.replace('tree_gen', 'results')
        # Also make sure to get rid of the .yaml at the end
        if top_write_dir[-5:] == '.yaml':
            top_write_dir=top_write_dir[:-5]
    write_config_tree(sweep_config_dir, top_write_dir)
    print(f"Successfully wrote config files to {top_write_dir}")

def write_config_tree(sweep_config_dir, top_write_dir):
    # Load in sweep config with base parameters and sweep parameters
    sweep_config = load_config(sweep_config_dir)
    # Expand the keys in the config - ['key1:key2:key3'] to ['key1']['key2']['key3']
    expand_keys(sweep_config)

    # Consolidate base parameters with all sweep parameters into a "tree" where each leaf is a unique parameter set
    consolidated_dict = consolidate_parameters(sweep_config['parameter_dicts'], addtl_list=[sweep_config['base_dict']])

    # # Consolidate all of the sweep parameters into a "tree" where each leaf is a unique parameter set
    # consolidated_dict = consolidate_parameters(sweep_config['parameter_dicts'])

    # Create a dictionary where each key is a directory and the value is the unique parameter set
    directory_dict = create_directory_dict(consolidated_dict, path_len=len(sweep_config['parameter_dicts'])-1)
    # # I'm curious if I make sure each value is deepcopied whether it will resolve my issue with ~commands
    # directory_dict = {k: deepcopy(v) for (k,v) in directory_dict.items()} # Answer: no (not on its own at least)
    # Now each value is a full set of parameters, including the base parameters
    # expand_directory_dict(directory_dict, sweep_config['base_dict'])

    # Write each set of parameters as a config file
    write_directory_dict(directory_dict, top_write_dir)

def contractuser(path: str):
    home_dir = os.path.expanduser('~')
    if path.startswith(home_dir):
        return path.replace(home_dir, '~')
    return path

def overwrite_configs(configs_directory, overwrite_directory):
    pass
    # TODO: write this function
    # config_dirs = get_config_dirs(ctop_dir = configs_directory)
    # configs = [load_config(dir_) for dir_ in config_dirs]
    # overwrite_config = expand_keys(load_config(overwrite_directory))

    # new_configs = []
    # for config, dir_ in zip(configs, config_dirs):
    #     new_config = merge_dicts()

    # pass
