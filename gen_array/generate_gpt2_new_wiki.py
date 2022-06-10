import os

temperature_list = [0.1, 0.5, 0.9]
top_p_list = [0.5, 0.7, 0.9, 0.95]
top_k_list = [20, 50, 100, 500]
# model_list = ['gpt3_curie', 'gpt3_davinci', 'gpt3_ft', 'gpt2', 'bart_ft', 't5_ft', 'gpt_neo']
model_list = ['gpt2']

input_dir = '/scratch/yd2481/NLG_evaluation/wiki_new.json'
output_dir = './gen_gpt2_new_wiki/'

idx = 0
for model in model_list:
    for temperature in temperature_list:
        for top_p in top_p_list:
            idx += 1
            command = '''#!/bin/bash
#
#SBATCH --job-name=generation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

if [[ "$(hostname -s)" =~ ^g[r,v] ]]; then nv="--nv"; fi

cd /scratch/yd2481
singularity exec $nv \
--bind /vast/yd2481/cache:$HOME/.cache --overlay overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate eval
cd /scratch/$USER/NLG_evaluation
python /scratch/yd2481/NLG_evaluation/main.py -g -m {model} -p {input_dir} --top_p {top_p} --temperature {temperature} -o {output_dir}
"
            '''.format(top_p = top_p, temperature = temperature, model = model, input_dir = input_dir, output_dir = output_dir)

            output_file_path = os.path.join(output_dir, str(idx) + '.s')
            print(output_file_path)
            with open(output_file_path, 'w') as f:
                f.write(command)


        
        for top_k in top_k_list:
            idx += 1
            command = '''#!/bin/bash
#
#SBATCH --job-name=generation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

if [[ "$(hostname -s)" =~ ^g[r,v] ]]; then nv="--nv"; fi

cd /scratch/yd2481
singularity exec $nv \
--bind /vast/yd2481/cache:$HOME/.cache --overlay overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate eval
cd /scratch/$USER/NLG_evaluation
python /scratch/yd2481/NLG_evaluation/main.py -g -m {model} -p {input_dir} --top_p {top_p} --temperature {temperature} -o {output_dir}
"
            '''.format(top_p = top_p, temperature = temperature, model = model, input_dir = input_dir, output_dir = output_dir)
            output_file_path = os.path.join(output_dir, str(idx) + '.s')


            with open(output_file_path, 'w') as f:
                f.write(command)