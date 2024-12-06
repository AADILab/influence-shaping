I'm storing information related to results here.

## 10_29_2024

This is the first bunch of experiments related to influence based reward shaping using the rovers and uavs environment. Each experiment (or set? of experiments?) has a name associated with it according to the order in which the experiments were generated. The experiments are in alphabetical order where lower letters in the alphabet (a,b,c) were generated first. If each experiment was described just a letter, that would be boring, so each experiment has a name where the first letter is the corresponding letter of the alphabet.

### jet

I want to make rovers even more dependent on uavs. Now the rovers have a small observation radius of 5 and cannot sense any POIs. They must rely entirely on their uav sensors (and technically they can also still use their rover sensors) to be guided in close coordination to a POI. Hopefully this helps make D-Indirect stand out as the clear winner when rovers absolutely have to depend on uavs for help and they must remain nearby in order to recieve that help. Remaining nearby is key here because that's what the influence heuristic will use to reward uavs for helping the rovers.

### kitkat

I want to play around a bit with the idea of seperating rovers and uavs into clearly seperated pairs. One of the issues (I suspect) with D-Indirect not working very effectively is the idea that D-Indirect is tracking throughout the episode what rover was with which uav. And if all the rovers and uavs start in one big clump in the middle of the map, then it makes it hard to seperate out who is influencing who. The idea here is to put one rover-uav pair on one corner of the map and another rover-uav pair in another corner of the map. Then the influence counters starting racking up very clearly as high for one rover-uav pair and high for the rover-uav pair. So there shouldn't be much of a tossup of who is influencing who. Each uav starts already influencing one rover. And each rover can only see with an observation radius of 5 and they can't see POIs. So the rover's behavior (quite literally based on the fact that its sensors at the beginning of the episode can only sense the neary uav) is coupled to the uav it spawns next to.
