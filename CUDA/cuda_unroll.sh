#!/bin/bash
#SBATCH --job-name=reduce	      	## Name of the job.
#SBATCH --output=reduce_unroll.out		## Output log file
#SBATCH --error=reduce_unroll.err		## Error log file
#SBATCH -A class-eecs224-gpu    	## Account to charge
#SBATCH -p gpu              		## run on the gpu partition
#SBATCH -N 1                		## run on a single node
#SBATCH -n 1                		## request 1 task
#SBATCH -t 00:15:00         		## 15-minute run time limit
#SBATCH --gres=gpu:V100:1   		## request 1 gpu of type V100

# Module load Cuda Compiler
module load cuda/10.1.243

# Runs a bunch of standard command-line
# utilities, just as an example:

echo "Script began:" `date`
echo "Node:" `hostname`
echo "Current directory: ${PWD}"

echo ""
echo "=== Running 5 trials of unroll ... ==="
for trial in 1 2 3 4 5; do
  echo "*** Trial ${trial} ***"
  ./unroll
done

echo ""
echo "=== Done! ==="

# eof
