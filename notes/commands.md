This is for storing commands for using commandline tools included in this repo to generate figures for papers and so forth.

Using commit hash abe111e1c1f6398f6731bcde951f18a662751202

# PhD Dissertation

## Performance Curves

Generating the performance curves for learning in stochastic settings (1x1, 2x2, 3x3, 4x4).

```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/36_fix_sweep/1x1 --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/36_fix_sweep/1x1/comparison.phd.pdf --ylim 0 1.1 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo
```
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/36_fix_sweep/2x2 --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/36_fix_sweep/2x2/comparison.phd.pdf --ylim 0 3.25 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo
```
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/36_fix_sweep/3x3 --title "" -o /nfs/stak/users/gonzaeve/influence-
shaping/outfigs/2025_10_17/alpha/36_fix_sweep/3x3/comparison.phd.pdf --ylim 0 6 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo
```
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/36_fix_sweep/4x4 --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/36_fix_sweep/4x4/comparison.phd.pdf --ylim 0 9.25 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo
```

Generating performance curves for rover-passing ([1 rover / 4 drones] and [2 rovers /8 drones]).
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/41_4uavhallway --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/41_4uavhallway/comparison.phd.pdf --ylim 0 3 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo --legend-loc "lower right"
```
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/43_4uavhallway_times_2uavs --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/43_4uavhallway_times_2uavs/comparison.phd.pdf --ylim 0 5 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo --legend-loc "lower right"
```

Generating performance curves for the team archive ablation study ([4x4 and [1 rover / 4 drones]]).

```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/45_editing_ablation_study_figures/2x2 --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/45_editing_ablation_study_figures/2x2/comparison.phd.pdf --ylim 0 3.25 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo --legend-
loc "upper left"
```
```
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2025_10_17/alpha/45_editing_ablation_study_figures/hallway_1 --title "" -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2025_10_17/alpha/45_editing_ablation_study_figures/hallway_1/comparison.phd.pdf --ylim 0 3 --ylabel "POIs Captured" --window_size 50 --fitness-colors --legend-order acm-telo --legend-loc "lower right"
```
