To plot bar graphs for jaamas paper:

Single Lane:
1 POI:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/1_poi -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/1_poi/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 1.1 --showbest --generation 2500

2 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/2_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/2_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 2.2 --showbest --generation 20000

3 POIs:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/15_single_lane/3_pois/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/15_single_lane/3_pois/bar.pdf --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 3.3 --showbest --generation 50000

2 Lanes:
4 POIs (2 rovers), initial attempt:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/11_2_lanes/2_pois_per_lane/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/11_2_lanes/2_pois_per_lane/bar.png --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 5000

4 POIs (2 rovers), 0pt spacing:
clear && python tools/plot/single_bar_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/16_double_lane/0_spacing/ -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/16_double_lane/0_spacing/bar.png --window 500 --ylabel "POIs Captured" --yticks 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 --axes-position 0.1 0.3 0.89 0.57 --fitness-colors --figsize 8 6 --labelmap jaamas-all --xtick-rotation 60 --grouping jaamas-split --dpi 300 --ylim 0 4.4 --showbest --generation 50000

To plot generations comparison:
python tools/plot/single_gens_comparison.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/13_5x5_tmp --dpi 300 -o /nfs/stak/users/gonzaeve/influence-shaping/outfigs/2026_05_06/13_5x5_tmp/gens_comparison.png --log-scale --fitness-colors --ylim 1 200000 --xlim -0.2 3.2 --yticks 10 100 1000 10000 100000 --ylabel "Num. Generations Required" --methods influence-extension --xlabel  "Number of Drones in Chain" --labelmap influence-extension --exclude D-Indirect-Window-N6-n0 D-Indirect-Window-N5-n0 D-Indirect-Window-N4-n0 --legend-facecolor "lightgray" --legend-labelspacing 1.8

Learning Curves:
1 rover, 3 drones, 1 POI

python tools/plot/comparisons.py /nfs/stak/users/gonzaeve/influence-shaping/results/2026_05_06/17_jaamas_edits/single_chain/1_poi --xlim 0 3000 --window 500 --fitness-colors --no-legend --marker-outline --dpi 300 --yticks 0 0.5 1.0 --xticks 0 1000 2000 3000 --ylabel "POIs Captured" --xlabel "Num. Generations" --figsize 3 4 --num-markers 3 --axes-position 0.21 0.15 0.74 0.8 --xticklabels 0 1K 2K 3K --ylim -0.1 1.1
