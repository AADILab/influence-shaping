To plot bar graphs for jaamas paper:

2 POIs:

clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/2_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/2_pois/bar.png --generation 20000 --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 2.1

3 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/3_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/3_pois/bar.png --generation 20000 --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 --axes-pos
ition 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi
 300 --ylim 0 3.1
