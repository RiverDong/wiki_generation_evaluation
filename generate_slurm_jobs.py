import os
from constants import *
from utils import *

wiki_prompt_dict = {'old': '/scratch/yd2481/NLG_evaluation/wiki_old.json', 'new':  '/scratch/yd2481/NLG_evaluation/wiki_new.json'}
output_dir = '/scratch/yd2481/NLG_evaluation/generation'

## GPT2 with different sampling parameters and temperatrues
GPT2_ablation_top_p(wiki_prompt_dict, output_dir)

GPT2_ablation_top_k(wiki_prompt_dict, output_dir)

## All models
all_models(wiki_prompt_dict, output_dir)



