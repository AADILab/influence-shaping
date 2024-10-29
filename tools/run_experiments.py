from ccea_lib import runCCEA

print("ccea_lib")
config_dir = "~/influence-shaping/results/preliminary/may_20/stabilize_learning/debug_a/config.yaml"
runCCEA(config_dir=config_dir)

from ccea_toolbox import runCCEA

print("ccea_toolbox")
config_dir = "~/influence-shaping/results/preliminary/may_20/stabilize_learning/debug_b/config.yaml"
runCCEA(config_dir=config_dir)
