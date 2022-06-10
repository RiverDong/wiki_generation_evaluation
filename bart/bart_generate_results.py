# from utils import *
import json
import time
# from utils import *
from wiki.models import *
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

model_params={
    "MODEL":"/scratch/yd2481/wiki/bart/bart_fine_tune/model_files",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":2,          # training batch size
    "VALID_BATCH_SIZE":2,          # validation batch size
    "TRAIN_EPOCHS":2,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":100,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":512,   # max length of target text
    "SEED": 42                     # set seed for reproducibility 
}


out_file_path = '/scratch/yd2481/wiki/generation_results/bart_generation.json'
num_responses = 20

train_data = pd.read_csv('/scratch/yd2481/wiki/wiki_data/wiki_fine_tune.csv')
test_data = pd.read_csv('/scratch/yd2481/wiki/wiki_data/wiki_test.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BartTokenizer.from_pretrained(model_params["MODEL"])
model = BartForConditionalGeneration.from_pretrained(model_params["MODEL"])
model = model.to(device)


for idx, row in test_data.iterrows():
    dic = {}
    dic['prompt'] = row['prompt']
    dic['wiki_completion'] = row['completion']

    input_ids = tokenizer(row['prompt'], return_tensors="pt").input_ids
    outputs = model.generate(input_ids.to(device, dtype = torch.long), 
                    max_length=500, 
                    do_sample=True, 
                    top_k = 50,
                    num_return_sequences=20
                    )

    dic['t5'] = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in outputs]
    with open(out_file_path, 'a') as f:
        f.write(json.dumps(dic) + "\n")