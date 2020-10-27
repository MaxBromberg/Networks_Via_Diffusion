#!/bin/bash
file_directory="/home/mpib/bromberg/Networks_Via_Diffusion"
for sub in $(seq -w 0 32) ; do
	echo '#!/bin/bash'					                    > job.slurm
	echo "#SBATCH --job-name Maqz_$sub"			        >> job.slurm
	echo "#SBATCH --mem 16GB"				                >> job.slurm
	echo "#SBATCH --cpus-per-task 20"			          >> job.slurm
	echo "#SBATCH --workdir /home/mpib/bromberg/"		>> job.slurm
 	echo "module load conda && conda activate EchoChambers && python3 $file_directory/grid_search_control.py $sub" 							 >> job.slurm
	sbatch job.slurm
	rm -f job.slurm
done

