#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/run_test-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

module load python
conda activate fgas-ml

echo "run.py"

srun python -u run.py --config-data configs/data/config.yaml --config-run configs/run/mlp_regressor.yaml