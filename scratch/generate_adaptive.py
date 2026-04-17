# Basically this is code to generate yaml files for my adaptive influence experiments
# Let's add parameters where we can put empty spaces (no POIs) in between spaces that have a drone and POI
# One parameter is how many POIs there are
# Other parameter is how big the spaces are (integer value)

# Each space is 20x20
# We add 25 timesteps for each additional space
# Default starting point is just 1 rover, 1 drone, 1 POI in 20x20 space.
# Make the rover stay within observation radius of the drone
# POIs are opposite of the drone in their space

import argparse
import yaml

def is_divisible(divisor: int, dividend: int)->bool:
    return divisor % dividend == 0

def is_even(num: int)->bool:
    return is_divisible(num, 2)

def is_odd(num: int)->bool:
    return not is_divisible(num, 2)

def is_gap_space(idx: int, gap_size: int)->bool:
    # If gap size is 1, then every 2nd space has a POI
    # 0: gap, 1: no gap, 2: gap, 3: no gap, 4: gap, 5: no gap
    # If gap size is 2, then every 3rd space has a POI
    divisor = idx+1
    dividend = gap_size+1
    if is_divisible(divisor, dividend):
        return False
    return True

def is_poi_space(idx: int, gap_size: int)->bool:
    return not is_gap_space(idx=idx, gap_size=gap_size)

def generate_space(idx: int, gap_size: int) -> dict:
    space_dict = {}
    # start means this is the first space
    # so place a rover
    if idx==0:
        rover = {
            'action': {
                'type': 'dxdy'
            },
            'needs_uav_to_move': True,
            'observation_radius': 5.0,
            'policy': {
                'type': 'network'
            },
            'position': {
                'fixed': [
                    5.0,
                    5.0
                ],
                'spawn_rule': 'fixed'
            },
            'resolution': 90,
            'sensor': {
                'accum_type': 'sum',
                'type': 'RoverLidar'
            },
            'reward_spec': {
                'reward_type': 'Global'
            }
        }
        space_dict['rovers'] = [rover]

    # Then place a uav based on parity and space
    # bounds based on the space number
    size = 20
    low_x = idx*size
    high_x = (idx+1)*size
    pos_x = idx*size+5
    if is_odd(idx):
        pos_y=15.0
    else:
        pos_y=5.0
    uav = {
        'action': {
            'type': 'dxdy'
        },
        'bounds': {
            'high_x': high_x,
            'low_x': low_x,
            'high_y': 20,
            'low_y': 0
        },
        'observation_radius': 100.0,
        'position': {
            'fixed': [
                pos_x,
                pos_y
            ],
            'spawn_rule': 'fixed'
        },
        'resolution': 90,
        'reward_spec': {
            'reward_type': 'Global'
        }
    }
    space_dict['uavs'] = [uav]
    # # Then check if we have a POI in this space
    if is_poi_space(idx=idx, gap_size=gap_size):
        pos_x = idx*size+15
        if is_odd(idx):
            pos_y=5.0
        else:
            pos_y=15.0
        poi = {
            'capture_radius': 10.0,
            'constraint': 'sequential',
            'coupling': 1,
            'dissapear_bool': True,
            'observation_radius': 5.0,
            'position': {
                'fixed': [
                    pos_x,
                    pos_y
                ],
                'spawn_rule': 'fixed'
            },
            'value': 1.0
        }
        space_dict['hidden_pois'] = [poi]

    return space_dict

def compute_total_spaces(num_pois: int, gap_size: int)->int:
    return num_pois*(gap_size+1)

def compute_sequence_params(num_pois: int, gap_size: int)->dict:
    total_spaces = compute_total_spaces(num_pois=num_pois, gap_size=gap_size)
    x_size = total_spaces*20
    y_size = 20
    num_steps = total_spaces*25
    sequence_dict = {
        'hidden_pois': [],
        'uavs': [],
        'rovers': [],
        'num_steps': num_steps,
        'map_size': [
            x_size,
            y_size
        ]
    }
    for idx in range(total_spaces):
        space_dict = generate_space(idx=idx, gap_size=gap_size)
        for key in ['rovers', 'uavs', 'hidden_pois']:
            if key in space_dict:
                for item in space_dict[key]:
                    sequence_dict[key].append(item)
    return sequence_dict

def generate_config_snippet(num_pois: int, gap_size: int)->dict:
    # Figure out where rovers, uavs, pois go
    sequence_dict = compute_sequence_params(num_pois=num_pois, gap_size=gap_size)
    config = {
        'env': {
            'agents': {},
            'pois': {
                'rover_pois': []
            }
        }
    }

    config['env']['agents']['rovers'] = sequence_dict['rovers']
    config['env']['agents']['uavs'] = sequence_dict['uavs']
    config['env']['pois']['hidden_pois'] = sequence_dict['hidden_pois']
    config['env']['map_size'] = sequence_dict['map_size']
    config['ccea'] = {'num_steps': sequence_dict['num_steps']}
    return config

def test():
    # Testing
    print(' --- test is_gap_space() ---')
    print('gap_size = 0 | All outputs should be false.')
    gap_size = 0
    for idx in range(10):
        out=is_gap_space(idx=idx, gap_size=gap_size)
        print(idx, ' | ', out)
    print('gap_size=1 | 0,2,4,6,8 should be true')
    gap_size = 1
    for idx in range(10):
        out=is_gap_space(idx=idx, gap_size=gap_size)
        print(idx, ' | ', out)
    print('gap_size=2 | 0,1,3,4,6,7,9 should be true')
    gap_size = 2
    for idx in range(10):
        out=is_gap_space(idx=idx, gap_size=gap_size)
        print(idx, ' | ', out)
    print(' --- test compute_total_spaces() --- ')
    num_pois = 1
    for gap_size in range(3):
        total_spaces = compute_total_spaces(num_pois=num_pois, gap_size=gap_size)
        print(f'num_pois: {num_pois} | gap_size: {gap_size} | total_spaces: {total_spaces}')
    gap_size = 3
    for num_pois in [1,2,3]:
        total_spaces = compute_total_spaces(num_pois=num_pois, gap_size=gap_size)
        print(f'num_pois: {num_pois} | gap_size: {gap_size} | total_spaces: {total_spaces}')

def main():
    parser = argparse.ArgumentParser(description="Generate adaptive influence experiment configuration.")
    parser.add_argument('--num_pois', type=int, required=True, help='Number of POIs')
    parser.add_argument('--gap_size', type=int, required=True, help='Gap size (number of empty spaces between POIs)')
    parser.add_argument('--output', type=str, default=None, help='Output file (optional, prints to stdout if not set)')
    args = parser.parse_args()

    # Call sequence generator function with parsed arguments
    result = generate_config_snippet(num_pois=args.num_pois, gap_size=args.gap_size)

    # Output result to file or print
    if args.output:
        with open(args.output, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
    else:
        print(yaml.dump(result, default_flow_style=False))

if __name__=='__main__':
    main()
