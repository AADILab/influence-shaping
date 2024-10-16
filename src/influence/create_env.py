from librovers import rovers, thyme, std, eigen
import os

"""
This is a quick prototype of a simple rover domain.
I want 4 rovers with 4 POIs
The map is 10x10
The coupling is 1
Rovers just need to spread out and go the POIs
Rovers start in the center
POIs are positioned in the corners

Rovers have 8 sensors total (4 POI sensors, 4 rover density sensors)
Rovers learn using G or D (this should be a commandline option or
    something similar)
POIs have an observation radius of 5 or something like that

"""

# Create rovers
Dense = rovers.Lidar[rovers.Density]
Discrete = thyme.spaces.Discrete
num_rovers = 4
agents = [rovers.Rover[Dense, Discrete](1.0, Dense(90)) for _ in range(num_rovers)]

# Create pois
num_pois = 4
coupling = 1
pois = [rovers.POI[rovers.CountConstraint](1) for _ in range(num_pois)]

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
basic_yaml_path = source_dir = os.path.abspath(os.path.join(current_dir,"configs", "basic.yaml"))

agent_positions = [
    [1.0, 1.0],
    [4.0, 5.0],
    [5.0, 5.0],
    [5.0, 4.0]
]

poi_positions = [
    [1.0, 9.0],
    [1.0, 1.0],
    [9.0, 9.0],
    [9.0, 1.0]
]

Env = rovers.Environment[rovers.CustomInit]
env = Env(rovers.CustomInit(agent_positions, poi_positions), agents, pois)

# I think env is using copies of these lists so we need to reset them
# explicitly like this
env.set_rovers(agents)
env.set_pois(pois)

# env.reset() does not reset rover positions. It calls the rover's reset function
# which only resets the internal path of the rover. And it sets all POIs to unobserved
states, rewards = env.reset()
# Actually I think it does reset the rover and POI positions according to the setup/.hpp file
# In this case it is according to setup/init_corners.hpp

print("Sample environment state (each row corresponds to the state of a rover): ")
for state in states:
    # print(state)
    print(type(state))
    print(state[0])
    print(state.transpose())

for r in env.rovers():
    print(r.position().x, r.position().y)

for p in env.pois():
    print(p.position().x, r.position().y)

# Now that I have the environment set up and (somewhat) understand how to modify it,
# let's try throwing some learning at this thing

