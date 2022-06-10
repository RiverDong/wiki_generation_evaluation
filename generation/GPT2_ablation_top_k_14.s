#!/bin/bash
#
#SBATCH --job-name=generation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

if [[ "$(hostname -s)" =~ ^g[r,v] ]]; then nv="--nv"; fi

cd /scratch/yd2481
singularity exec $nv --bind /vast/yd2481/cache:$HOME/.cache --overlay overlay-50G-10M.ext3:ro /scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash -c "
source /ext3/env.sh
conda activate eval
cd /scratch/$USER/NLG_evaluation
python /scratch/yd2481/NLG_evaluation/main.py -g -m gpt2 -p /scratch/yd2481/NLG_evaluation/wiki_new.json --top_p 1 --top_k 50 --temperature 0.1 -o /scratch/yd2481/NLG_evaluation/generation
"
    