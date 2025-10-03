find /nfs/stak/users/gonzaeve/influence-shaping/results/10_01_2025/alpha/4_replicate_gecco/ -name config.yaml | while read cfg; do
    echo "Running with $cfg"
    python tools/run/config.py "$cfg" --load
done
