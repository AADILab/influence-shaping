rsync

rsync redback redback_no_checkpoints -avr --exclude **pkl --dry-run

rsync is a tool for copying files around in linux. It can be used to copy files across machines (may be worth switching to for hpc).

cut -d "," -f 3-4 eval_team_0_joint_traj.csv 

cut is like cat, but you can specify only to print out certain columns
This command uses the "," as a delimter to only print the 3rd-4th columns of a csv file
Useful for quickly checking things about joint trajectories in the terminal
