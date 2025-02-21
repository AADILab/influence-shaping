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

This one will just be a 1_rover_1_uav/random_pois_10x10 experiment where we compare G and D-Indirect using the same seed for random number generation. The results for G and D-Indirect should look exactly the same... And it turns out they do look exactly the same. This successfully demonstrated that the only differences in performance between G and D-Indirect came from the randomness of mutations and selection, nothing to do with the fitness shaping method itself.

**NOTE:** I double checked the config for these experiments and the uav observation radius is set properly to 1000.0

### tackle

Time to do a big sweep of different experiments playing around with map size (this includes number of pois), numbers of rovers, and uavs. The idea here is to get a good lay of the land of what is going on with sequentially observable pois with the small observation radius of rovers, large capture radius for pois, and comparing D-Indirect with static influence vs dynamic influence.

**NOTE:** I discovered an issue with these results. These results are supposed to be testing D-Indirect with dynamic influence, but were run with an out-of-date version of the rovers submodule that does not support timestep based credit for D-Indirect. So likely the timestep based D-Indirect results are actually using the previous D-Indirect that is trajectory based.

### ultra

tackle is taking too long (and was run with outdated code). This is going to be a smaller subset of those experiments with less generations and less timesteps.

Looking through the results, it seems like there isn't much difference in performance between any method when we're using when there is only 1 rover. I suppose that makes sense. I wouldn't expect credit assignment to help out a whole lot when there is only 1 rover capapble of observing pois.

It's a different story when there are 3 rovers. When there are 3 rovers and 1 uav, G is clearly the way to go. That's kind of strange given that there are 3 rovers and that means 3 agents can go around directly receiving credit. You'd think you need credit assignment between rovers so the one that learned to follow the uav does so... but maybe they all learn to follow the uav.

When there are 3 rovers and 3 uavs, credit assignment still doesn't do much. For some reason, D-Indirect-Timestep-System (which I added as a baseline that I expected to perform poorly) is the only method that is able to perform on par with G. No credit assignment is actually doing anything here.

I have two ideas for what to test moving forward.
1: More timesteps. I realized that when I looked at previous experiments I ran similar to this one, there were more timesteps. When we look at the results for ultra, we see that the teams hit (at best) a little under 2 pois on average.

When we look at quartz/1_rover/1_uav/random_pois_50x50, we see the team hits on average a bit over 2 pois. And the trends for performance actually look like they are going up, not just hitting a point and plateauing.

2: Small capture radius for pois. If the global reward is dense enough, you don't need credit assignment, unless you have a TON of agents. In my case, I only have a few agents. So if I add some sparsity then I should see credit assignment start to matter more again.

**NOTE:** There is an issue with these results. The observation radius for uavs is 5.0 rather than 1000.0 so these uavs have a super limited observation radius, which is different from what I expected when interpretting results.

### vector

This is the ultra experiments but this time with more timesteps. Instead of 50, we're going to use 150.

**NOTE:** There is an issue with these results. The observation radius for uavs is 5.0 rather than 1000.0.

Performance is definitely better than when timesteps are 50, but I should re-run this (or some variant of this) with the larger observation radius for uavs. It's a bit impressive that the teams can learn even though the uavs can't see very far.

Looking at the learned joint trajectories, it looks like the best performing team (D-Indirect-Step trial 0) learned to trace out a big triangle to maximize the odds of capturing pois. That makes sense given that the pois spawn randomly across the entire map. Maybe would have made more sense to learn a square or spiral or something, but the capapbilities are also quite limited because the only thing that changes from timestep to timestep in the observations (for the most part) are the rover and uav observing each other moving around. And only when they are within 5 units of each other.

G trial 6 showed a team learning a kind of line... when uavs cant see very far ahead, then we see this kinds of shape tracing behaviors. Worth noting: in this G trial the uav does not stick with the rover the whole time, it seems like d-indirect-step does effectively help uavs stick with rovers whereas G does not provide the same incentive (uav still gets credit for rover's actions even if the uav is far away from the rover, whereas with D-i-step the uav only gets credit for the rover's actions if the uav is nearby the rover.)

### wumbo

This is the vector experiments but with a small capture radius for the pois. Capture radius of 5.0 rather than 1000.0

**NOTE:** There is an issue with these results. The observation radius for uavs is 5.0 rather than 1000.0.

These results look quite similar to vector. I suppose that is to be expected - uavs can't see much. I just looked at the high-level comparison plots.

### xylophone

This is the vector experiments but with the larger observation radius for the uavs. The 1 rover 1 uav experiment should closely match the result in quartz/1_rover_1_uav/random_pois_50x50. Only 5 trials because the HPC queue is nearly full.

I'm having a hard time breaking G. D is easy to break here, but G is difficult and I can't tell exactly why. I suppose it may be coming from some noise from the influence heuristic but I'm not really sure. The 1 rover 1 uav experiment closely matches the quartz 1 rover 1 uav experiment on visual inspection. 

### yellow

This is the wumbo experiments but with the larger observation radius for uavs. We should see fitness shaping (I think all D and D-Indirect variants) show a better performance gain here compared to the xylophone experiments where pois have a large capture radius. In this case, rewards are far sparser, so reward shaping should be more important.

It seems like in these experiments compared to xylophone the performance of D-Indirect-Timestep System and G are closer to each other. I get the sense that there is definitely something here to making capture radius smaller making fitness shaping more important, even if the immediate effect of that is making (what I thought was) the weakest fitness shaping outperform everything else.

I think moving forward I should keep the sparsity from small capture radius. I'm also thinking I should investigate further experiments that use 1 to 1 rover to uav OR multiple uavs per rover. If I do multiple rovers per uav then I don't think D-Indirect-Timestep will help much. Dynamic influence is for when we have multiple uavs that need to support one rover but at different times... So I should play around with that a bit.

## 01_05_2025

### australia

This was an attempt to run a larger experiment where I introduced poi types. I set up 4 pairs of rovers+uavs (4 rovers, 4 uavs, paired 1-1). I divided a 100,100 map into 4 squares (2x2 smaller squares), and put one uav-rover pair at the center of each square.

I did 3 variants of this. In the first variant, I used standard pois (20 of them) that spawn randomly (uniform) across the full map. In the second, I used 4 poi types (A,B,C,D), where each one corresponds to a type that one of the uavs can sense. Uav 0 could sense [A], Uav 1 could sense [B], etc. In the third, those pois of different types were constrained to spawn in the subsquare that matched the uav with the sensing capability to sense that poi.

The results are quite messy. It seems like in the standard 20 pois case the rovers and uavs cannot effectively learn to coordinate and instead just try to spread out to maximize their coverage of the space. G only gets up to about 5 pois on average, and that's the highest performing method. D is the lowest (as expected), and the different D-Indirect or mixed fitness shaping variants land somewhere in between G and D.

After explaining the results to the lab, I was suggested to try a few things in order to debug what's going on.

1. Increase observation radius of rovers. The idea is that they should not have a hard time seeing uavs and that their limited observation radius may make it too difficult for them to find a uav to follow.

2. Increase network size. 12 inputs to 10 hidden units does not seem right. Try increasing the number of hidden units, and this might make it possible to learn better policies.

3. Make a rover policy where it follows the nearest uav. It would use a large observation radius to almost always see a uav. And I can play with what to do when it does not see a uav. (I am thinking to just have the rover sit still. It's up to the uavs to find and direct the rover.)

4. Change the rover's action to just choosing which uav to follow. (The idea is to simplify this problem so that the rovers don't have to learn that they have to follow uavs, just make it so that the rovers automatically follow the uavs and they are just choosing who to follow.)

I'm not sure this line of experiments will yeild much (but I will set them all up). I am instead thinking that setting up atomic tests are the best way to go. In other words, take the smallest version of this problem, explore it, optimize it, then scale it back up. I think that's the best way to get a handle on what's going on.

### brisbane

This is the first atomic test (actually ran this before the lab discussion of australia results). I did two configs. One config was a rover-uav pair discovering 5 pois. The rover has a small observation radius (5 units) but spawns next to the uav. The pair is able to get up to about 2.5 pois on average after 1,000 generations with G. This suggests to me that the small observation radius of rovers should not be an issue. This was also with 5 hidden pois so the rover is entirely dependent on the uav.

In the other variant, I designed a fully independent rover that has a large observation radius (1,000 units aka the entire map) and the pois are not hidden from the rover. They are normal pois. The rover is able to get a slightly higher performance on average using G after 1,000 generations. The average is about 3 pois. It learns really fast compared to the much more pronounced learning curve when we have 1 rover and 1 uav.

### cassowary

This is a large set of atomic tests with 1 rover and 1 uav. I wanted to fiddle with a lot of different parameters to see if any of them could learn to get all 5 pois, or generally just perform better than other variants.

Here are the parameters I played with
timesteps: 50, 100, 150
standard deviation for mutating weights: 0.1, 0.5, 0.9
independent probability of mutating weights: 0.1, 0.5, 0.9
density sensor vs exponential sensor for rovers and uavs
uav max velocity: 1.0, 2.0
rover observation radius: 5.0, 1000.0

### dingo

Same as cassowary, but with 3 pois. (I realized I could queue up more experiments to the HPC and number of pois had been another parameter on my mind I wanted to experiment with.)

### emu

I'm just running here steps 1 and 2 of exploring the australia results.

1. Increase obervation radius of rovers (5.0, 50.0, 100.0)
2. Increase network size (10, 15, 20)

Just running with the standard pois, no need to play with different poi types yet.
NOTE: I am not rerunning the combination of parameters that gives us the original standard pois experiment in australia. I am going to populate the results for that combination using the original australia experiment results. That saves a bunch of computation and gets me these results faster.

### emu_subset
NOTE: I cancelled a lot of emu experiments to run a much smaller subset because running all of them was taking too long to run. I just ran observation radius 50 + network size 10, and observation radius 50 + network size 15.

### falcon

More atomic experiments. Looking at:

- 1, 2, 3 pois
- 20, 50, 100, 150 timesteps
- 5 1000 observation radius for rover
10 trials

I'm hoping I can find a small atomic experiment that starts with a low G but hits the high score after 1,000 generations. Then I can scale that up and presumably that problem will really benefit from credit assignment. There is too much going on with the bigger experiments to know what is going wrong from just looking at the results.

### galah

Atomic experiments, now with 2 rovers, 2 uavs, and 2 pois. For some reason G is still winning here. I am running a version of this where the rover can see the pois, no uavs, and the rovers have a large observation radius. In that case, I should at least see D outperform G.

### galah_follow

galah experiments but rovers follow the nearest uav (without learning)

### galah_independent_rovers

galah experiments but now with rovers that are fully capable of getting to pois on their own. I am hoping to see D outperform G here. If not, I need to do some investigating. And maybe take out the hall of fame evaluation. I think that is helping too much.

G outperforms D. Why? Credit assignment should be helping a whole lot here. It's just rovers that need to learn to each get the pois near where they individually spawn. Also for some reason D-Indirect-Timestep is performing equivalently to G even though it should be equivalen to D... Not sure why that is happening but I don't think it's worth debugging right now. right now I want to see D outperform G one way or another... Maybe I can do a version of this where rovers have small observation radii and are faaaar away from each other in that case D should really work.

I think the thing that is messing up credit assignment is that the rovers are close enough to see each other as input to their networks and that is causing them to act differently based on who their partner is. So you can't isolate them very well.

### galah_independent_rovers_seperate
Rovers have much smaller observation radii and are really far away from one another.

### hammerhead

This is completing the four squares experiments that Kagan asked for Tues Jan 7, 2025. One variant where rovers learn to follow uavs and one variant where rovers are preprogrammed to follow.

### irukandji

This is re-running some experiments from emu that seem to have given me some conflicting results I'm investigating. Hopefully by running these again I can get some clarity on what is going on.

### xmasbeetle

Lots of debugging here

### yabby

This is where I'm using the new selection mechanisms that Gaurav and I made. N elites with team elites and elite preservation.

#### 02

This one had a mistake in it. The map size is 50x50, but needs to be 150x150, so all rovers and uavs are immediately shoved into one corner because of the way the bounds are used.

#### 04

Redo of 02.

3x3 random circle poi experiments. 1,000 generations. G and D-Indirect-Timestep (the all or nothing variant). Subpopulation size of 50 individuals

There are 3 different versions of this running that do different things with the elites
1. standard_n_elites - This uses a standard n-elites selection mechanism. Take the top 5 best individuals (based on individual fitness) at this generation and put them into the offspring for this subpopulation. Then fill in the rest with a binary tournament
2. preserve_n_elites - This takes elite preservation to its logical extreme. This preserves 5 teams (5 individuals based on team fitness) and 5 individuals (5 individuals based on individual fitness). Preservation means that these individuals are not evaluated at each generation. They can be knocked out of preservation status by a better team or individual during selection.  This also takes the top 5 best individuals and top 5 best teams at this generation and puts them into the offspring for the subpopulation. Then the rest of the offspring is filled in with a binary tournament. Any elite (preserved or otherwise) can be selected into the tournament. No elites (preserved or otherwise) are mutated during the mutation step.
3. use_agg_team_fit - This does all of the elite preservation, with the caveat that teams are not sorted by team fitness. Instead, they are sorted based on the sum of all individual fitnesses for that team. So if a team is 2 rovers, 2 uavs, and each rover got a poi, and one uav got credit for a poi, then the individual fitnesses look like this:
[ 1, 1, 1, 0 ]. Aggregated fitness is 3. G would be 2. This changes how elite teams are selected, like you were describing last night
3. best_combo_prediction - My personal prediction of what will work the best. 5 elite teams and 5 elite individuals. No elite preservation. Teams are sorted based on aggregated fitness in each team, not based on that team's fitness.
