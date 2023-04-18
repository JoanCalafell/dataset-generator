#!/bin/bash
#SBATCH --job-name=INTERP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1932
#SBATCH --time=48:00:00
###SBATCH --qos=debug
###SBATCH --exclusive
###SBATCH --qos=bsc_case

module purge
module load intel/2017.4 impi/2017.4 mkl/2017.4
module load python/3.7.4 
module load hdf5/1.8.19

echo "EXECUTING	PLANE INTERPOLATOR"
srun python plane_interp.py 
echo 'EXECUTION DONE'
