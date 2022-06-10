import openai
import pandas as pd
import numpy as np
import json
import time
import transformers
import torch
import os
import textstat
import spacy
import nltk 
import random
from nltk import tokenize
import matplotlib.pyplot as plt
from scipy import stats
from gensim.models import KeyedVectors
from keras.models import load_model
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
from constants import *

## CONSTANTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

class generator:
    def __init__(self, args, model_name):
        self.read_file_path = args.prompt
        self.num_prompt = args.num_prompt
        self.model_name = model_name
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature
        self.num_return_sequences = args.num_return_sequences
        self.max_length = args.max_length
        self.wiki_timestamp = 'new' if 'new' in args.prompt else 'old'
        output_file_name = model_name + '_top-p_' + str(self.top_p) + '_top-k_' + str(self.top_k) + '_temp_' + str(self.temperature) + '_' + self.wiki_timestamp + '.json'
        self.out_file_path = os.path.join(args.output_dir, output_file_name)
        openai.key = args.openai_key

    def load_model(self):
        print(self.model_name)
        if self.model_name == 'gpt_neo':
            tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
            model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
        elif self.model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        elif self.model_name == 'bart_ft':
            model_path = "./bart/bart_fine_tune/model_files"
            tokenizer = BartTokenizer.from_pretrained(model_path)
            model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
        elif self.model_name == 't5_ft':
            model_path = "./t5/model_files"
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        return tokenizer, model
        print('{model} not yet implemented'.format(model = self.model_name))
    
    def generate_json(self):
        if os.path.exists(self.out_file_path):
            return 
        if 'gpt3' in self.model_name:
            self.generation_gpt3()
        else:
            self.generation_free()

    def generation_free(self):
        if os.path.exists(self.out_file_path):
            return
        tokenizer, model = self.load_model()
        f = open(self.read_file_path)
        for idx, line in enumerate(f):
            if idx % 100:
                print(idx)
            if idx >= self.num_prompt:
                break

            line = json.loads(line)
            first_sent, rest_para = split_into_sentences(line['text'])
            prompt = line['title'] + '\n' + first_sent

            dic = {}
            dic['prompt'] = prompt
            dic[self.model_name] = self.generation_free_helper(prompt, tokenizer, model)
            dic['wiki_completion'] = line['text']
            with open(self.out_file_path, 'a') as f:
                f.write(json.dumps(dic) + "\n")

    def generation_free_helper(self, prompt, tokenizer, model):
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        res = []
        outputs = model.generate(inputs,
                                max_length=self.max_length, 
                                do_sample=True, 
                                top_k = self.top_k,
                                top_p = self.top_p,
                                temperature = self.temperature,
                                num_return_sequences=self.num_return_sequences
                                )
        for i in range(self.num_return_sequences):
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if prompt not in text:
                text += prompt 
            res.append(text)
        return res


    def generation_gpt3_helper(self, prompt, engine):
        num_ans_per_request = 10
        res = []
        num_request_times = self.num_response_per_prompt // num_ans_per_request

        for i in range(num_request_times):
            try:
                response = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        top_k = self.top_k,
                        top_p = self.top_p,
                        temperature = self.temperature,
                        stop=["\n"],
                        max_tokens=self.max_length,
                        n = num_ans_per_request
                        )
                text = [response['choices'][i]['text'] for i in range(num_ans_per_request)]
                for t in text:
                    if prompt not in t:
                        t = prompt + t
                    res.append(t)
                time.sleep(5)
            except:
                print('GPT3: failed to connect, wait for 30 seconds')
                time.sleep(30)
        return res

    def generation_gpt3(self):
        if os.path.exists(self.out_file_path):
            return

        if self.model_name == 'gpt3_curie':
            engine = 'curie'
        elif self.model_name == 'gpt3_davinci':
            engine = 'davinci'
        elif self.model_name =='gpt3_ft':
            engine = 'curie:ft-new-york-university-2021-12-06-06-35-17'

        f = open(self.read_file_path)
        for idx, line in enumerate(f):
            line = json.loads(line)
            if idx % 100 == 0:
                print(idx)
            if idx >= self.num_prompt:
                break

            first_sent, rest_para = split_into_sentences(line['text'])
            prompt = line['title'] + '\n' + first_sent

            dic = {}
            dic['prompt'] = prompt
            dic[self.model_name] = self.generation_gpt3_helper(prompt, engine)
            dic['wiki_completion'] = line['text']
            
            with open(self.out_file_path, 'a') as f:
                f.write(json.dumps(dic) + "\n")

def get_model_list(model_name):
    if model_name == 'all':
        return model_name_list
    elif model_name == 'all_free':
        return [model for model in model_name_list if 'gpt3' not in model]
    elif model_name in model_name_list:
        return [model_name]
    else:
        raise Exception('Model name not invalid')

def parse_title(file_name):
    # eg: gpt3_ft_top-p_2_top-k_2_temp_2_new.json
    model, top_p, top_k, temperature = file_name.rstrip('.json').replace('_top-p_', '|').replace('_top-k_', '|').replace('_temp_', '|').replace('_new', '').replace('_old', '').split('|')
    wiki_timestamp = 'new' if 'new' in file_name else 'old'
    return model, top_p, top_k, temperature, wiki_timestamp

def join_json_to_csv(output_dir):
    combined_df = pd.DataFrame()
    for generation_file in os.listdir(output_dir):   
        model, top_p, top_k, temp = parse_title(generation_file)
        generation_file = os.path.join(output_dir, generation_file)
        df = pd.read_json(generation_file, lines=True)
        print(generation_file, df.columns)

        df = df.rename(columns = {model: 'generation'})
        df['model'] = model
        df['top_p'] = top_p
        df['top_k'] = top_k
        df['temperature'] = temp

        df = df.explode('generation')
        df = df.dropna()
        combined_df = pd.concat([combined_df, df])
    return combined_df


def split_into_sentences(text):
    tmp = tokenize.sent_tokenize(text)
    return tmp[0], ' '.join(tmp[1:])

## Below are functions copied from essay scoring github repo
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model[i])        
    vec = np.divide(vec,noOfWords)
    return vec


def getVecs(essays, word2vec_model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, word2vec_model, num_features)
        c+=1
    return essay_vecs

def essay_scoring(text):
    word2vec_model = KeyedVectors.load_word2vec_format('./essay_scoring/word2vecmodel.bin', binary=True)
    lstm_model = load_model('./essay_scoring/final_lstm.h5')
    testing_vectors = getVecs([text], word2vec_model, 300)
    testing_vectors = np.array(testing_vectors)
    testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))
    score = lstm_model.predict(testing_vectors)
    return score[0][0]

def essay_scoring_list(text):
    word2vec_model = KeyedVectors.load_word2vec_format('./essay_scoring/word2vecmodel.bin', binary=True)
    lstm_model = load_model('./essay_scoring/final_lstm.h5')
    testing_vectors = getVecs(text, word2vec_model, 300)
    testing_vectors = np.array(testing_vectors)
    testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))
    score = lstm_model.predict(testing_vectors)
    return np.squeeze(score)

## Below are functions are evaluation metrics
def s_bert_score(sentence_list_1, sentence_list_2):
    s_bert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    embeddings1 = s_bert_model.encode(sentence_list_1, convert_to_tensor=True).to(device)
    embeddings2 = s_bert_model.encode(sentence_list_2, convert_to_tensor=True).to(device)
    score = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
    score = [score[i,i] for i in range(len(score[0]))]
    return score

    # s_bert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    # res = []
    # batch_size = 256
    # num_batch = (len(sentence_list_1) // batch_size) + 1
    # print(num_batch)

    # for i in range(num_batch):
    #     print(i, device)
    #     start_index = i * batch_size
    #     end_index = max((i+1) * batch_size, len(sentence_list_1) - 1)
    #     batch_list_1 = sentence_list_1[start_index : end_index]
    #     batch_list_2 = sentence_list_2[start_index : end_index]

    #     embeddings1 = s_bert_model.encode(batch_list_1, convert_to_tensor=True).to(device)
    #     embeddings2 = s_bert_model.encode(batch_list_2, convert_to_tensor=True).to(device)

    #     for idx in range(len(embeddings1)):
    #         score = util.cos_sim(embeddings1[idx], embeddings2[idx])
    #         res.append(score)
    # return res

def info_density(text):
    if text == '':
        return 0
    return len(set(nlp(text).ents)) / word_count(text)

def calc_relevance(wiki, generation):
    if wiki == '' or generation == '':
        return 0
    wiki_ents = set([str(x) for x in nlp(wiki).ents])
    generation_ents = set([str(x) for x in nlp(generation).ents])
    overlap_count = 0

    for ent in wiki_ents:
        if ent in generation_ents:
            overlap_count += 1
    if len(generation_ents) == 0:
        return 0
    relevance = overlap_count / len(generation_ents)
    return relevance

def flesch_reading_ease(text):
    return textstat.textstat.flesch_reading_ease(text)


def gunning_fog(text):
    if text == '':
        return 0
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade

# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
	# return textstatistics().syllable_count(word)
    return textstat.syllable_count(word)    

# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return textstat.textstat.legacy_round(ASPW, 1)

# Return total Difficult Words in a text
def difficult_words(text):

    # nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]

    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)

def break_sentences(text):
        # nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)

# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words

# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)


def get_hist(processed_df, eval_metric):
    hist, bins = np.histogram(processed_df[eval_metric], bins=50, normed=True)
    bin_centers = (bins[1:]+bins[:-1])*0.5
    return hist, bin_centers

def get_result(df, eval_metrics):
    stats, bins_dict = {}, {}
    df['length_by_char'] = df['generation'].apply(lambda x: len(x)).astype(int)
    df['length_rank'] = df.groupby(['model', 'prompt'])["length_by_char"].rank("dense", ascending=False).astype(int)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    processed_df = df.select_dtypes(include=numerics)
    processed_df = processed_df[processed_df['length_rank'] <= 9]
    processed_df = processed_df[processed_df['length_by_char'] != 0]

    for col in ['length_by_char','length_rank','info_density', 'gunning_fog','flesch','relevance', 'essay_scoring']:
        q_low = processed_df[col].quantile(0.001)
        q_hi  = processed_df[col].quantile(0.999)
        processed_df = processed_df[(processed_df[col] <= q_hi) & (processed_df[col] >= q_low)]
    
    for eval_metric in eval_metrics:
        hist, bin_centers = get_hist(processed_df, eval_metric)
        bins_dict[eval_metric] = [hist, bin_centers]
        stats[eval_metric] = [processed_df[eval_metric].mean(), processed_df[eval_metric].std(), processed_df[eval_metric].median()]
    return stats, bins_dict

def get_distribution(df):
    model_list = set(df['model'].tolist())
    mean_table, std_table = pd.DataFrame(), pd.DataFrame()
    eval_metrics = list(set(eval_metrics_dict.keys()) & set(df.columns))

    grouped = df.groupby(['model', 'top_p', 'top_k', 'temperature'])
    for group_attribute, group_df in grouped:
        stats, bins = get_result(group_df, eval_metrics)
        mean_dic = {'model': group_attribute[0], 'top_p': group_attribute[1], 'top_k': group_attribute[2], 'temperature': group_attribute[3]}
        std_dic = {'model': group_attribute[0], 'top_p': group_attribute[1], 'top_k': group_attribute[2], 'temperature': group_attribute[3]}

        for metric, score in stats.items():
            mean_dic[metric], std_dic[metric] = score[0], score[1]
        # mean_table = mean_table.append(pd.Series(mean_dic,  index=df.columns, name=model), ignore_index=True)[eval_metrics]
        mean_table = mean_table.append(mean_dic, ignore_index=True)
        std_table = std_table.append(std_dic, ignore_index=True)
    return mean_table, std_table


eval_metrics_dict = {
    'flesch': flesch_reading_ease,
    'essay_scoring': essay_scoring,
    'relevance': calc_relevance,
    's_bert': s_bert_score,
    'info_density': info_density,
    'gunning_fog': gunning_fog,
    # 'spcificity': get_specificity
}

def eval_one_file(args, file_name):
    output_file_path = os.path.join(args.output_dir, file_name.replace('json', 'csv'))
    if os.path.exists(output_file_path):
        return

    model, top_p, top_k, temp, wiki_timestamp = parse_title(file_name)
    generation_file = os.path.join(args.output_dir, file_name)
    df = pd.read_json(generation_file, lines=True)
    if len(df) < 2000:
        os.system('rm {generation_file}'.format(generation_file = generation_file))
        print(generation_file + 'is removed because it\'s partially generated')
    print(generation_file, df.columns)

    df = df.rename(columns = {model: 'generation'})
    df['model'] = model
    df['top_p'] = top_p
    df['top_k'] = top_k
    df['temperature'] = temp
    df['wiki_timestamp'] = wiki_timestamp

    df = df.explode('generation')
    df = df.dropna().reset_index(drop=True)

    for eval_metric_name, eval_metric_function in eval_metrics_dict.items():
        print('start evaluation of ', eval_metric_name)
        if eval_metric_name == 'essay_scoring':
            df['essay_scoring'] = essay_scoring_list(df['generation'].tolist())
            df['wiki_essay_scoring'] = essay_scoring_list(df['wiki_completion'].tolist())
        elif eval_metric_name == 's_bert':
            sentence_list_1, sentence_list_2 = df['wiki_completion'].tolist(), df['generation'].tolist()
            df['s_bert'] = s_bert_score(sentence_list_1, sentence_list_2)
        elif eval_metric_name == 'relevance':            
            df[eval_metric_name] = df[['wiki_completion', 'generation']].parallel_apply(lambda x: eval_metric_function(*x), axis = 1)
        elif eval_metric_name == 'flesch' or eval_metric_name == 'gunning_fog' or eval_metric_name == 'info_density':    
            df[eval_metric_name] = df['generation'].parallel_apply(eval_metric_function)
            df['wiki_' + eval_metric_name] = df['wiki_completion'].parallel_apply(eval_metric_function)
    df.to_csv(output_file_path)

    ## None paralleled version for debug purpose
    
    # for eval_metric_name, eval_metric_function in eval_metrics_dict.items():
    #     print('start evaluation of ', eval_metric_name)
    #     if eval_metric_name == 'essay_scoring':
    #         df['essay_scoring'] = essay_scoring_list(df['generation'].tolist())
    #     elif eval_metric_name == 's_bert':
    #         sentence_list_1, sentence_list_2 = df['wiki_completion'].tolist(), df['generation'].tolist()
    #         df['s_bert'] = s_bert_score(sentence_list_1, sentence_list_2)
    #     elif eval_metric_name == 'relevance':            
    #         df[eval_metric_name] = df[['wiki_completion', 'generation']].apply(lambda x: eval_metric_function(*x), axis = 1)
    #     elif eval_metric_name == 'flesch' or eval_metric_name == 'gunning_fog' or eval_metric_name == 'info_density':    
    #         df[eval_metric_name] = df['generation'].apply(eval_metric_function)
    # df.to_csv(output_file_path)


## Used to create wiki_data_old_3000.json
def random_select_from_wiki_dump(wiki_folder, output_path):
    # wiki_folder = '/scratch/yd2481/wikiextractor/text'
    folder_list = os.listdir(wiki_folder)
    # output_path = '/scratch/yd2481/wiki/wiki_data_old_3000.json'

    count = 0
    while count < 3000:
        target_folder = random.choice(folder_list)
        file_list = os.listdir(os.path.join(wiki_folder, target_folder))
        file = random.choice(file_list)
        target_file = os.path.join(wiki_folder, target_folder, file)
        print(target_file)
        f = open(target_file)
        for line in f:
            line = json.loads(line)
            if len(line['text'].split(' ')) > 300:
                dic = {"title": line["title"], "text": line["text"]}
                with open(output_path, 'a') as o:
                    o.write(json.dumps(dic) + "\n")
                break
        count += 1
    return

## Used to create fine-tune txt for specificity model 
def create_wiki_sent_txt(wiki_folder, output_path):
    # wiki_folder = '/scratch/yd2481/wikiextractor/text'
    folder_list = os.listdir(wiki_folder)
    # output_path = '/scratch/yd2481/wiki/wiki_data_old_3000.json'

    count = 0
    while count < 3000:
        target_folder = random.choice(folder_list)
        file_list = os.listdir(os.path.join(wiki_folder, target_folder))
        file = random.choice(file_list)
        target_file = os.path.join(wiki_folder, target_folder, file)
        
        print(target_file)
        f = open(target_file)
        for line in f:
            line = json.loads(line)
            if len(line['text'].split(' ')) > 300:
                sents = nlp(line['text']).sents
                sents = [sent for sent in sents]
                sent = str(random.choice(sents)).strip()
                if sent != '':
                    with open(output_path, 'a') as o:
                        o.write(json.dumps(sent) + "\n")
                    break
        count += 1
    return 

def copy_all_generation_file(root_path, new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    root_path = '/scratch/yd2481/NLG_evaluation'
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path) and folder.startswith('gen_'):
       
            for file in os.listdir(folder_path):
                if file.endswith('json'):
                    file_path = os.path.join(root_path, folder, file)
                    wiki_timestamp = 'new' if 'new' in folder_path else 'old'
                    output_file = file.rstrip('.json') + '_' + wiki_timestamp + '.json'
                    output_path = os.path.join(new_dir, output_file)
                    command = 'cp {file_path} {output_path}'.format(file_path = file_path, output_path = output_path)
                    os.system(command)
                    
def opt():
    from transformers import OPTModel, OPTConfig
    configuration = OPTConfig()
    model = OPTModel(configuration)
    configuration = model.config
    from transformers import GPT2Tokenizer, OPTModel
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
    model = OPTModel.from_pretrained("facebook/opt-350m")
    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    generation = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(generation)


def get_command(top_p, top_k, temperature, model, input_path, output_dir):
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
python /scratch/yd2481/NLG_evaluation/main.py -g -m {model} -p {input_path} --top_p {top_p} --top_k {top_k} --temperature {temperature} -o {output_dir}
"
    '''.format(top_p = top_p, top_k = top_k, temperature = temperature, model = model, input_path = input_path, output_dir = output_dir)
    return command

def GPT2_ablation_top_p(wiki_prompt_dict, output_dir):
    model = 'gpt2'
    top_k = 50
    idx = 0
    task_name = 'GPT2_ablation_top_p'
    
    for input_path in wiki_prompt_dict.values():
        for temperature in temperature_list:
            for top_p in top_p_list:
                idx += 1
                command = get_command(top_p, top_k, temperature, model, input_path, output_dir)
                output_file_path = os.path.join(output_dir, task_name + '_' + str(idx)+ '.s')
                print(output_file_path)
                with open(output_file_path, 'w') as f:
                    f.write(command)

def GPT2_ablation_top_k(wiki_prompt_dict, output_dir):
    model = 'gpt2'
    top_p = 1
    idx = 0
    task_name = 'GPT2_ablation_top_k'

    for input_path in wiki_prompt_dict.values():
        for temperature in temperature_list:
            for top_k in top_k_list:
                idx += 1
                command = get_command(top_p, top_k, temperature, model, input_path, output_dir)
                output_file_path = os.path.join(output_dir, task_name + '_' + str(idx)  + '.s')
                print(output_file_path)
                with open(output_file_path, 'w') as f:
                    f.write(command)

def all_models(wiki_prompt_dict, output_dir):
    temperature=0.9,
    top_p=1,
    top_k = 50
    idx = 0
    task_name = 'all_models_generation'

    for model in model_name_list:
        for input_path in wiki_prompt_dict.values():
            idx += 1
            command = get_command(top_p, top_k, temperature, model, input_path, output_dir)
            output_file_path = os.path.join(output_dir, task_name + '_' + str(idx)  + '.s')
            print(output_file_path)
            with open(output_file_path, 'w') as f:
                f.write(command)
