import random
import pandas as pd
from pandarallel import pandarallel
import torch
import argparse
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('NLG Evaluation', add_help=False)

    ## Requried arugments
    parser.add_argument('-e', '--evaluation', action='store_true')
    parser.add_argument('-g', '--generation', action='store_true')
    parser.add_argument('-p', '--prompt', default = './wiki_newly_released_3000.json', type = str,
            help = 'path the prompt json file')


    parser.add_argument('-o', '--output_dir', default = './generation', type=str)
    parser.add_argument('-n', '--num_prompt', default = 2000, type=int)
    parser.add_argument('--num_return_sequences', default = 20, type=int)
    parser.add_argument('--max_length', default = 400, type=int)
    parser.add_argument('-m', '--model_name_list', default = 'all_free', type=str, 
            help = 'all_free generates for all except gpt3, all generates for all including gpt3, can also specify')

    parser.add_argument('--openai_key', default = "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", type=str)

    ## Generation Sampling Params
    parser.add_argument('--top_p', default = 1, type=float)     # []
    parser.add_argument('--top_k', default = 20, type=int)     # [20, 50, 100, 500]
    parser.add_argument('--temperature', default = 0.9, type=float) # [0.1, 0.5, 0.9]
    return parser

def generation(args):
    model_name_list = get_model_list(args.model_name_list)
    print('The list of models that will used for generation: ', model_name_list)
    for model_name in model_name_list:
        print('generating text for {model} with top_p {top_p} top_k {top_k} temperature {temperature}'.format(model = model_name, top_p = args.top_p, top_k = args.top_k, temperature = args.temperature))
        # try:
            # my_generator = generator(args.prompt, args.output_dir, args.num_prompt, model_name, args.top_p, args.top_k, args.temperature, args.openai)
        my_generator = generator(args, model_name)
        my_generator.generate_json()
        # except:
        #     print('failed to generate using {model}, please check model exists/GPT3 openai key is passed in as an argument.')


def evaluation(args):
    pandarallel.initialize(progress_bar=True)

    for generation_file in os.listdir(args.output_dir):
        print('start evaluating' + generation_file)
        eval_one_file(args, generation_file)
    
def combine_df(dir):    
    combined_df = pd.DataFrame()
    for generation_file in os.listdir(dir):
        if generation_file.endswith('csv'):
            print(generation_file)
            df = pd.read_csv(os.path.join(dir, generation_file))
            combined_df = pd.concat([combined_df, df])
    output_path = os.path.join(dir, 'final_generation.csv')
    combined_df.to_csv(output_path)


#     mean_tablce, std_table = get_distribution(combined_df)
#     mean_table_output_path = os.path.join(args.output_dir, 'mean.csv')
#     std_table_output_path = os.path.join(args.output_dir, 'std.csv')
#     mean_table.to_csv(mean_table_output_path)
#     std_table.to_csv(std_table_output_path)

#     print(mean_table.to_latex().replace("\\\n", "\\ \hline\n"))
#     print(std_table.to_latex().replace("\\\n", "\\ \hline\n"))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    args = get_args_parser()
    args = args.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.generation:
        print('start generation')
        generation(args)
    if args.evaluation:
        print('start evaluation')
        evaluation(args)







# def evaluation(args):
#     combined_df = pd.DataFrame()
#     pandarallel.initialize(progress_bar=True)
#     for generation_file in os.listdir(args.output_dir):   
#         model, top_p, top_k, temp = parse_title(generation_file)
#         generation_file = os.path.join(args.output_dir, generation_file)
#         df = pd.read_json(generation_file, lines=True)
#         print(generation_file, df.columns)

#         df = df.rename(columns = {model: 'generation'})
#         df['model'] = model
#         df['top_p'] = top_p
#         df['top_k'] = top_k
#         df['temperature'] = temp

#         df = df.explode('generation')
#         df = df.dropna()

#         for eval_metric_name, eval_metric_function in eval_metrics_dict.items():
#             print('start evaluation of ', eval_metric_name)
#             if eval_metric_name == 'essay_scoring':
#                 df['essay_scoring'] = essay_scoring_list(df['generation'].tolist())
#             elif eval_metric_name == 's_bert':
#                 sentence_list_1, sentence_list_2 = df['wiki_completion'].tolist(), df['generation'].tolist()
#                 df['s_bert'] = s_bert_score(sentence_list_1, sentence_list_2)
#             elif eval_metric_name == 'relevance':            
#                 df[eval_metric_name] = df[['wiki_completion', 'generation']].parallel_apply(lambda x: eval_metric_function(*x), axis = 1)
#             elif eval_metric_name == 'flesch' or eval_metric_name == 'gunning_fog' or eval_metric_name == 'info_density':    
#                 df[eval_metric_name] = df['generation'].parallel_apply(eval_metric_function)
#         combined_df = pd.concat([combined_df, df])
#     eval_path = os.path.join(args.output_dir, 'eval.csv')
#     combined_df.to_csv(eval_path)