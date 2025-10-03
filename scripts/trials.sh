CONFIG_DIR=/nfs/stak/users/gonzaeve/influence-shaping/results/10_01_2025/alpha/4_replicate_gecco/Global-no-preservation

for t in $(seq 0 199); do
    echo "Running trial $t for $CONFIG_DIR/config.yaml"
    python tools/run/config.py "$CONFIG_DIR/config.yaml" --load -t $t
done
