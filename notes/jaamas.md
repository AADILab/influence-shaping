# To plot bar graphs for jaamas paper:

## Single Lane:
### 1 POI:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/1_poi -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/1_poi/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 1.1 --showbest --generation 2500

### 2 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/2_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/2_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 2.2 --showbest --generation 20000

### 3 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/3_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/3_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 3.3 --showbest --generation 50000

## 2 Lanes:
### 4 POIs (2 rovers), initial attempt:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/11_2_lanes/2_pois_per_lane/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/11_2_lanes/2_pois_per_lane/bar.png --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 5000

### 4 POIs (2 rovers), 0pt spacing:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/16_double_lane/0_spacing/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/16_double_lane/0_spacing/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 50000

## 4 POIs (2 rovers), 10pt spacing:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/16_double_lane/10_spacing/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/16_double_lane/10_spacing/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 50000

# To plot generations convergence comparison:
python tools/plot/single_gens_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/13_5x5_tmp --dpi 300 -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/13_5x5_tmp/gens_comparison.pdf --log-scale --fitness-colors --ylim 1 200000 --xlim -0.2 3.2 --yticks 10 100 1000 10000 100000 --ylabel "Num. Generations Required" --methods influence-extension --xlabel  "Number of Drones in Chain" --labelmap influence-extension --exclude D-Indirect-Window-N6-n0 D-Indirect-Window-N5-n0 D-Indirect-Window-N4-n0 --legend-facecolor "lightgray" --legend-labelspacing 1.8 --axes-position 0.12 0.12 0.87 0.86 --window 100

# Learning Curves:
## 1 rover, 3 drones, 1 POI
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/1_poi --xlim 0 3000 --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --yticks 0 0.5 1.0 --xticks 0 1000 2000 3000 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 3 4 --num-markers 4 --axes-position 0.21 0.15 0.72 0.82 --xticklabels 0 1K 2K 3K --ylim -0.1 1.1 -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/1_poi/comparison.pdf

## 1 rover, 6 drones, 2 POIs
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/2_pois --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 3 4 --num-markers 5 --axes-position 0.21 0.15 0.72 0.82 --ylim -0.2 2.2 --xlim 0 20000 --xticks 0 5000 10000 15000 20000 --xticklabels 0 5K 10K 15K 20K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/2_pois/comparison.pdf

### PhD Presentation
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/2_pois --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 5 4 --num-markers 5 --ylim -0.2 2.2 --xlim 0 20000 --xticks 0 5000 10000 15000 20000 --xticklabels 0 5K 10K 15K 20K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/2_pois/comparison_presentation.png --axes-position 0.15 0.15 0.8 0.8 --dpi 500

## 1 rover, 9 drones, 3 POIs
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/3_pois --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 3 4 --num-markers 6 --axes-position 0.21 0.15 0.72 0.82 --ylim -0.3 3.3 --xlim 0 50000 --xticks 0 10000 20000 30000 40000 50000 --xticklabels 0 10K 20K 30K 40K 50K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/3_pois/comparison.pdf

### PhD Presentation
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/3_pois --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 5 4 --num-markers 6 --ylim -0.3 3.3 --xlim 0 50000 --xticks 0 10000 20000 30000 40000 50000 --xticklabels 0 10K 20K 30K 40K 50K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/3_pois/comparison_presentation.png --axes-position 0.15 0.15 0.8 0.8 --dpi 500

## 2 rovers, 12 drones, 4 POIs (10spacing)
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain/1-10spacing --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 4.5 4 --num-markers 6 --axes-position 0.12 0.15 0.83 0.82 --ylim -0.4 4.4 --xlim 0 50000 --xticks 0 10000 20000 30000 40000 50000 --xticklabels 0 10K 20K 30K 40K 50K -o /nfs/stak/u
sers/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain/1-10spacing/comparison.pdf

### PhD Presentation
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain/1-10spacing --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 5 4 --num-markers 6 --axes-position 0.12 0.15 0.83 0.82 --ylim -0.4 4.4 --xlim 0 50000 --xticks 0 10000 20000 30000 40000 50000 --xticklabels 0 10K 20K 30K 40K 50K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain/1-10spacing/comparison.png

## 2 rovers, 12 drones, 4 POIs (0spacing, or NO spacing)
python tools/plot/single_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain/2-nospacing --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 4.5 4 --num-markers 6 --axes-position 0.12 0.15 0.83 0.82 --ylim -0.4 4.4 --xlim 0 50000 --xticks 0 10000 20000 30000 40000 50000 --xticklabels 0 10K 20K 30K 40K 50K -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain/2-nospacing/comparison.pdf

# Multibar Comparison w. Norm POI Captures:
## Single Lane:
python tools/plot/single_multibar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain.pdf --fitness-colors --normalize-yscores --ylim 0 1.1 --ylabel "Norm. POI Capture Rate" --legend-order jaamas  --bar-order jaamas --legend-loc 'lower right' --xticklabels "3 Drones, 1 POI" "6 Drones, 2 POIs" "9 Drones, 3 POIs" --axes-position 0.08 0.1 0.9 0.88 --figsize 8 3 --no-legend

## Double Lane:
python tools/plot/single_multibar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain.pdf --fitness-colors --normalize-yscores --ylim 0 1.1 --ylabel "Norm. POI Capture Rate" --legend-order jaamas  --bar-order jaamas --legend-loc 'lower right' --axes-position 0.08 0.1 0.9 0.88 --figsize 8 3 --no-legend --xticklabels "Separated Chains" "Interfering Chains" --generation 50000

# Joint Trajectories:
## 1 rover, 3 drones, 1 POI
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj_edit.pdf --ylim -1 21 --xlim -1 61 --yticks 0 20 --xticks 0 20 40 60 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.89 0.8

python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj_edit.pdf --ylim -5 25 --xlim -5 65 --yticks 0 20 --xticks 0 20 40 60 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.7 0.8 --dpi 500 --rover-observation-radius

python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/1_poi/D-Indirect-Window-N4-n0/trial_1/gen_20000/test/team_0_joint_traj_edit.svg --ylim -5 25 --xlim -5 65 --yticks 0 20 --xticks 0 20 40 60 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.7 0.8 --dpi 500 --rover-observation-radius

## 1 rover, 6 drones, 2 POIs
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/2_pois/D-Indirect-Window-N4-n0/trial_0/gen_20000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/2_pois/D-Indirect-Window-N4-n0/trial_0/gen_20000/test/team_0_joint_traj_edit.svg --ylim -5 25 --xlim -5 125 --yticks 0 20 --xticks 0 20 40 60 80 100 120 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.89 0.8

## 1 rover, 9 drones, 3 POIs
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/3_pois/D-Indirect-Window-N4-n0/trial_0/gen_50000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/3_pois/D-Indirect-Window-N4-n0/trial_0/gen_50000/test/team_0_joint_traj_edit.pdf --ylim -5 25 --xlim -5 185 --yticks 0 20 --xticks 0 20 40 60 80 100 120 140 160 180 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.89 0.8

python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/3_pois/D-Indirect-Window-N4-n0/trial_0/gen_50000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/single_chain/3_pois/D-Indirect-Window-N4-n0/trial_0/gen_50000/test/team_0_joint_traj_edit.svg --ylim -5 25 --xlim -5 185 --yticks 0 20 --xticks 0 20 40 60 80 100 120 140 160 180 --use-image --individual-colors --ylabel "Y Position" --xlabel "X Position" --figsize 6 2.5 --axes-position 0.1 0.21 0.89 0.8

## 2 rovers, 12 drones, 6 POIs, 10 spacing
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain/1-10spacing/D-Indirect-Window-N4-n0/trial_4/gen_50000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain/1-10spacing/D-Indirect-Window-N4-n0/trial_4/gen_50000/test/team_0_joint_traj.svg --xticks 0 20 40 60 80 100 120 --yticks 0 20 30 50 --ylim -5 55 --xlim -5 125 --individual --rover-observation-radius --xlabel "X Position" --ylabel "Y Position" --figsize 6 5

## 2 rovers, 12 drones, 6 POIs, no spacing
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/double_chain/2-nospacing/D-Indirect-Window-N4-n0/trial_4/gen_50000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/17_jaamas_edits/double_chain/2-nospacing/D-Indirect-Window-N4-n0/trial_4/gen_50000/test/team_0_joint_traj.svg --xticks 0 20 40 60 80 100 120 --yticks 0 20 30 50 --ylim -5 55 --xlim -5 125 --individual --rover-observation-radius --xlabel "X Position" --ylabel "Y Position" --figsize 6 5

## 1 rover, 4 drones, 1 POI
python tools/plot/single_joint_trajectory.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/13_5x5_tmp/gap_size_3/D-Indirect-Window-N3-n0/trial_1/gen_20000/test/team_0_joint_traj.csv -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/13_5x5_tmp/gap_size_3/D-Indirect-Window-N3-n0/trial_1/gen_20000/test/team_0_joint_traj.svg --ind --figsize 6 2.5 --rover-observation-radius --ylabel "Y Position" --xlabel "X Position" --ylim -5 25 --xlim -5 85 --axes-position 0.1 0.1 0.7 0.8

# Animations
## 1 Rover, 1 Drone, 1 POI
### Curvy Trajectory
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/36_fix_sweep/1rover_1drone/trial_0/gen_1000/test/team_0_joint_traj_interpolated_edit.csv --use-image --xlim 20 40 --ylim 0 20 --yticks --xticks --icon-scale 0.5 --figsize 3 3 --axes-position 0.01 0.01 0.98 0.98 --dpi 2000

### Nice Trajectory
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/36_fix_sweep/1rover_1drone/trial_2/gen_1000/test/team_0_joint_traj_interpolated_edit_v2.csv --use-image --xlim 15 35 --ylim 10 30 --yticks --xticks --icon-scale 0.5 --figsize 3 3 --axes-position 0.01 0.01 0.98 0.98 --dpi 2000

## 4 Rovers, 4 Drones, 4 POIs
### Round Robin Trajectory
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/36_fix_sweep/4rovers_4drones/trial_0/gen_1000/test/team_0_joint_traj_interpolated.csv --use-image --xlim 10 90 --ylim 0 80 --yticks --xticks --icon-scale 0.7 --figsize 12 12 --axes-position 0.01 0.01 0.98 0.98

### Double Bottom Trajectory
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/36_fix_sweep/4rovers_4drones/trial_20/gen_500/test/team_0_joint_traj_interpolated.csv --use-image --xlim 10 90 --ylim 0 80 --yticks --xticks --icon-scale 0.7 --figsize 12 12 --axes-position 0.01 0.01 0.98 0.98

## 1 Rover, 4 Drones, 4 POIs
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/41_4uavhallway/D-Indirect-Timestep/trial_1/gen_5000/test/team_0_joint_traj_interpolated_corrected.csv --use-image --yticks --xticks 20 40 60 80 --icon-scale 0.5 --figsize 12 3 --axes-position 0.01 0.01 0.98 0.98 --dpi 1000 --individual --influence-shading

## 1 Rover, 6 Drones, 2 POIs
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/17_jaamas_edits/single_chain/2_pois/trial_0/gen_20000/test/team_0_joint_traj_interpolated_corrected.csv --use-image --yticks --xticks 20 40 60 80 100 --icon-scale 0.7 --figsize 18 3 --axes-position 0.05 0.05 0.9 0.9 --individual --influence-shading

## 2 Rovers, 12 Drones, 4 POIs
python tools/animate/single_joint_trajectory.py /Users/ever/influence-shaping/results/2026_06_09/01_video_edits/17_jaamas_edits/double_chain/1-10spacing/trial_1/gen_100000/test/team_0_joint_traj_interpolated_corrected.csv --use-image --yticks 20 30 --xticks 20 40 60 80 100 --icon-scale 0.7 --figsize 18 6 --axes-position 0.05 0.05 0.9 0.9 --individual --influence-shading
