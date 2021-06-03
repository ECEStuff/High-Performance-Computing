#!/bin/bash
#SBATCH --job-name=Mandelbrot      	## Name of the job.
#SBATCH --output=mandelbrot.out		## Output log file
#SBATCH --error=mandelbrot.err		## Error log file
#SBATCH -A class-eecs224     		## Account to charge
#SBATCH -p standard          		## Partition/queue name
#SBATCH --nodes=4           		## Number of nodes
#SBATCH --ntasks=160  			## Number of tasks (MPI processes)

# Module load boost
module load boost/1.71.0/gcc.8.4.0

# Module load MPI
module load mpich/3.4/intel.2020u1

# Run the program 
mpirun -np $SLURM_NTASKS ./mandelbrot_mw 10000 10000
