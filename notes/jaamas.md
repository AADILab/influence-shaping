To plot bar graphs for jaamas paper:

1 POI:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/1_poi -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/1_poi/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 1.1 --showbest --generation 2500

2 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/2_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/2_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 2.2 --showbest --generation 19700

3 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/3_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/3_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 3.3 --showbest --generation 16000

4 POIs (2 rovers):
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/11_2_lanes/2_pois_per_lane/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/11_2_lanes/2_pois_per_lane/bar.png --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 5000