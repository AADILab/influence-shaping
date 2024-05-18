from ccea_toolbox import runCCEA

# Ok let's do the easy one

folder_names = [
    # "00",
    # "1rover_2pois_1hidden_1normal",
    "1uav_1rover_2_pois_1hidden_1normal"
]

for name in folder_names:
    config_dir = "~/influence-shaping/results/preliminary/may_17/mutation_fix/"+name+"/config.yaml"
    runCCEA(config_dir=config_dir)
