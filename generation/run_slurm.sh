#! /bin/bash

for tmp in /scratch/yd2481/NLG_evaluation/generation/*.s;
    do  
        sbatch $tmp
    done