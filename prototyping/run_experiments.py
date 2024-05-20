from ccea_toolbox import runCCEA

# Ok let's do the easy one

# folder_names = [
#     # "1rover_1hiddenpoi",
#     # "1rover_1normalpoi",
#     "1uav_1rover_1hiddenpoi_moregens"
# ]

# for name in folder_names:
#     config_dir = "~/influence-shaping/results/preliminary/may_17/random_pois/"+name+"/config.yaml"
#     runCCEA(config_dir=config_dir)

config_dir = "~/influence-shaping/results/preliminary/may_18/save_joint_trajs/1uav_1rover_1hiddenpoi_100000gens/config.yaml"
runCCEA(config_dir=config_dir)
