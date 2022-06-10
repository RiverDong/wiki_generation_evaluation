#! /bin/bash

for tmp in /scratch/yd2481/NLG_evaluation/gen_array/gen_all_models_new_wiki.py/*.s;
    do  
        sbatch $tmp
    done


for tmp in /scratch/yd2481/NLG_evaluation/gen_array/gen_gpt2_old_wiki/*.s;
    do  
        sbatch $tmp
    done

for tmp in /scratch/yd2481/NLG_evaluation/gen_array/gen_all_models_old_wiki/*.s;
    do  
        sbatch $tmp
    done

for tmp in /scratch/yd2481/NLG_evaluation/gen_array/gen_gpt2_new_wiki/*.s;
    do  
        sbatch $tmp
    done
