from utils import *
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('NLG Evaluation', add_help=False)

    ## Requried arugments
    parser.add_argument('-e', '--evaluation', action='store_true')
    parser.add_argument('-g', '--generation', action='store_true')
    parser.add_argument('-p', '--prompt', default = './wiki_newly_released_3000.json', type = str,
            help = 'path the prompt json file')


    parser.add_argument('-o', '--output_dir', default = './generation_results', type=str)
    parser.add_argument('-n', '--num_prompt', default = 2000, type=int)
    parser.add_argument('-m', '--model_name_list', default = 'all_free', type=str, 
            help = 'all_free generates for all except gpt3, all generates for all including gpt3, can also specify')

    parser.add_argument('--openai_key', default = "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", type=str)

    ## Generation Sampling Params
    parser.add_argument('--top_p', default=1, type=float)     # []
    parser.add_argument('--top_k', default=50, type=int)     # [25, 100, 500]
    parser.add_argument('--temperature', default=0.9, type=float) # [0.1, 0.5, 0.9]
    return parser


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# args = get_args_parser()
# args = args.parse_args()

# my_generator = generator(args, 'gpt2')
# tokenizer, model = my_generator.load_model()
# res = my_generator.generation_free_helper('I try to test this code. ', tokenizer, model)
# print(res)

import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

class GenerationData(Dataset):
    def __init__(self, df, model_name):
        self.df = df.rename(columns = {model_name: 'generation'}).explode('generation').dropna()
    
    def __getitem__(self, index):
        prompt = self.df.iloc[index]['prompt']
        generation = self.df.iloc[index]['generation'].lstrip(prompt)
        wiki_completion = self.df.iloc[index]['wiki_completion'].lstrip(prompt)
        return {
            'text': '[CLS] ' + prompt + ' [SEP] ' +  generation + ' [SEP]',
            'wiki': False
        }, {
            'text': '[CLS] ' + prompt + ' [SEP] ' +  wiki_completion + ' [SEP]',
            'wiki': True
        }

file_path = '/scratch/yd2481/NLG_evaluation/generation_results/gpt2_top-p_1_top-k_50_temp_0.9.json'
df = pd.read_json(file_path, lines=True)
training_set = GenerationData(df, 'gpt2')
print(training_set.__getitem__(1))

# class SentimentData(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.text = dataframe.Phrase
#         self.targets = self.data.Sentiment
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.text)

#     def __getitem__(self, index):
#         text = str(self.text[index])
#         text = " ".join(text.split())

#         inputs = self.tokenizer.encode_plus(
#             text,
#             None,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             pad_to_max_length=True,
#             return_token_type_ids=True
#         )
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']
#         token_type_ids = inputs["token_type_ids"]


#         return {
#             'ids': torch.tensor(ids, dtype=torch.long),
#             'mask': torch.tensor(mask, dtype=torch.long),
#             'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
#             'targets': torch.tensor(self.targets[index], dtype=torch.float)
#         }