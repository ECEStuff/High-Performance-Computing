#!/bin/bash
#SBATCH --job-name=Mandelbrot      	## Name of the job.
#SBATCH --output=mandelbrot_serial.out		## Output log file
#SBATCH --error=mandelbrot_serial.err		## Error log file
#SBATCH -A class-eecs224     		## Account to charge
#SBATCH -p standard          		## Partition/queue name
#SBATCH --nodes=1           		## Number of nodes
#SBATCH --ntasks=1  			## Number of tasks (MPI processes)

# Module load boost
module load boost/1.71.0/gcc.8.4.0

# Module load MPI
module load mpich/3.4/intel.2020u1

# Run the program 
mpirun -np $SLURM_NTASKS ./mandelbrot_serial 1000 1000
