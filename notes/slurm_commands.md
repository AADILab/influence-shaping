### squeue

_squeue_

List all jobs on the system's queue

_squeue -u gonzaeve_

List all of my jobs on the queue

_squeue --Format=jobid,partition,name:50,state,timeused,timeleft,nodelist_

List all jobs in the queue according to the specified format

_squeue --Format=jobid,partition,name:50,state,timeused,timeleft,nodelist -u gonzaeve | grep \*\*australia\*\*_

Search all jobs (up to 50 characters in the name) for jobs that match the given regex expression

_squeue --Format=jobid,partition,name:50,state:20,timeused,timeleft,nodelist -u gonzaeve | grep RUNNING | wc -l_

Tell me how many of my jobs are currently running

_squeue --Format=jobid,partition,name:50,state:20,timeused,timeleft,nodelist -u gonzaeve | grep PENDING | wc -l_

Tell me how many of my jobs are still pending



squeue -w cn-v-[1-8] --Format=jobid:10,partition:10,name:70,state:10,timeused:10,timeleft:12,nodelist:10

watch -n0.1 squeue -u gonzaeve --Format=jobid:10,partition:10,name:70,state:10,timeused:10,timeleft:12,nodelist:10
