# Basically this is code to generate yaml files for my adaptive influence experiments
# Let's add parameters where we can put empty spaces (no POIs) in between spaces that have a drone and POI
# One parameter is how many POIs there are
# Other parameter is how big the spaces are (integer value)

# Each space is 20x20
# We add 25 timesteps for each additional space
# Default starting point is just 1 rover, 1 drone, 1 POI in 20x20 space.
# Make the rover stay within observation radius of the drone
# POIs alternate between being North and South (y=15 vs y=5)

# from enum import Enum

# class Parity(Enum):
#     ODD = 1
#     EVEN = 2

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
    if idx=0:
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

                ]
            }
        }

            # - capture_radius: 10.0
            #   constraint: sequential
            #   coupling: 1
            #   disappear_bool: true
            #   observation_radius: 5.0
            #   position:
            #     fixed:
            #     - 10.0
            #     - 5.0
            #     spawn_rule: fixed
            #   value: 1.0

    return space_dict

def generate_sequence(num_pois: int, gap_size: int)->dict:
    pass

if __name__=='__main__':
    # Testing
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
