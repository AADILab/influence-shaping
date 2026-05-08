# This should check a config and make sure it has everything required to run
# Also check if we have additional unused arguments and throw an error if so

# And I guess check for optional arguments and check within those optional arguments to make sure that has the required arguments, etc
# And then add a way to populate defaults for the optional arguments

top_params = {
    'ccea': dict,
    'env': dict,
    'experiment': dict,
    'processing': dict,
    'data': dict,
    'debug': dict
}

ccea_params = {
    'evaluation': dict,
    'mutation': dict,
    'network': dict,
    'num_generations': int,
    'num_steps': int,
    'populations': dict,
    'selection': dict,
    'team_formation': dict,
    'weight_initializations': dict
}
