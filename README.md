# Automatic Evaluation Metrics of Knowledge-intensive Long Text Generation

## Setup
```
cd wiki_generation_evaluation
pip install -r requirements.txt
```

## Generation
wiki_new.json and wiki_old.json are 
One can run the following to create one generation experiment.  
```
python main.py -g -m gpt2 -p wiki_old.json --top_p 0.7 --temperature 0.1 -o ./generation
```
In practice, we run all the experiments through slurm job on GPU cluster. The generation task is time-consuming and we do not suggest researchers without GPU cluster to redo the generation.

## Evaluation
```
python main.py -e -o /generation
```

## Results
This command gives a csv file called results.csv which is the final chart in our paper
