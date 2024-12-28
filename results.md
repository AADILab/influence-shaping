I'm storing information related to results here.

## 10_29_2024

This is the first bunch of experiments related to influence based reward shaping using the rovers and uavs environment. Each experiment (or set? of experiments?) has a name associated with it according to the order in which the experiments were generated. The experiments are in alphabetical order where lower letters in the alphabet (a,b,c) were generated first. If each experiment was described just a letter, that would be boring, so each experiment has a name where the first letter is the corresponding letter of the alphabet.

### jet

I want to make rovers even more dependent on uavs. Now the rovers have a small observation radius of 5 and cannot sense any POIs. They must rely entirely on their uav sensors (and technically they can also still use their rover sensors) to be guided in close coordination to a POI. Hopefully this helps make D-Indirect stand out as the clear winner when rovers absolutely have to depend on uavs for help and they must remain nearby in order to recieve that help. Remaining nearby is key here because that's what the influence heuristic will use to reward uavs for helping the rovers.

### kitkat

I want to play around a bit with the idea of seperating rovers and uavs into clearly seperated pairs. One of the issues (I suspect) with D-Indirect not working very effectively is the idea that D-Indirect is tracking throughout the episode what rover was with which uav. And if all the rovers and uavs start in one big clump in the middle of the map, then it makes it hard to seperate out who is influencing who. The idea here is to put one rover-uav pair on one corner of the map and another rover-uav pair in another corner of the map. Then the influence counters starting racking up very clearly as high for one rover-uav pair and high for the rover-uav pair. So there shouldn't be much of a tossup of who is influencing who. Each uav starts already influencing one rover. And each rover can only see with an observation radius of 5 and they can't see POIs. So the rover's behavior (quite literally based on the fact that its sensors at the beginning of the episode can only sense the neary uav) is coupled to the uav it spawns next to.

### lucky, monster, nitro, otter, pepper

Missing documentation on what all these were. need to go back to check

### quartz

I wanted to play around with having just one rover and one uav observing pois in a sequence. I'm using capture radius instead of observation radius for poi rewards so that rovers can get rewarded for "observing" pois even if they do not ever sense the pois. This should be easier than needing rovers to get within the observation radius of a poi (remember that means the poi must also be within the rover's obervation radius) in order for there to be a reward. The 1_rover_1_uav/random_pois_10x10 trials seemed to show something very strange, in that there is a statistically significant difference (based mean and standard error) between using D-Indirect and G for this problem. That does not make sense because numerically speaking, G and D-Indirect should always evaluate to the same exact number for these configurations. For some reason, G does much better than D-Indirect here.

### rabbit

The idea here is to help debug the seemingly anomolous result in **quartz**. The idea is to give pois a larger (or smaller) area in which to spawn in and see if there is a consistent difference between the performance of G or D-Indirect (even though, again, numerically speaking, D-Indirect and G should evaluate to the same number here). I also added a debug flag that will cause the ccea program to crash if at any point D-Indirect is computed for an agent and it is a different number than what G was computed as. (D-Indirect not equal to G crashes the ccea.)

Looking at the results, I think there is a general trend that the less random the pois are, the more the randomness in the co evolutionary process matters. There is more "luck" involved in getting the correct policy rather than a rigorous evaluation process ensure that the agents are learning robust policies. Take for instance the most extreme cases. On one end, we have fixed pois that never move. All you need to learn is a specific joint trajectory (and in fact, overfitting here is perfectly fine) and just do that. But if pois are instead completely random then the only policies that will thrive throughout training are the ones that can handle a variety of cases. What this is leading me to conclude is that my pois should be completely random for further experiments. Otherwise I risk seeing how well random mutations will lead to overfitting the environment rather than how well fitness shaping methods ensure we are learning robust joint policies.

Take a look at the results for 1_rover_1_uav/random_pois_10x10 for rabbit as opposed to quartz. There are different performances for G and D-Indirect. In rabbit the two lines are almost on top of each other, whereas there is more of a clean split in quartz. And the lines that are in rabbit look like they are in the middle of the lines of quartz. The point being: Randomness has a big role in determining what method comes out on top here. So I need to make sure I mitigate that moving forward.

### slide

One last thing I'd like to try. I want to fix the randomly generated numbers to see if G and D-Indirect consistently perform identically given the same random seed.

This one will just be a 1_rover_1_uav/random_pois_10x10 experiment where we compare G and D-Indirect using the same seed for random number generation. The results for G and D-Indirect should look exactly the same
