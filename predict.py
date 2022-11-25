from nmatheg import predict_from_run
import argparse
import os
from datasets import load_dataset
import json
# Create the parser
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--p', '-path', type = str, action='store') 
my_parser.add_argument('--n', '-num', type = int, action='store') 

args = my_parser.parse_args()
data_config = json.load(open(f"{args.p}/data/data_config.json"))
data = load_dataset(data_config["name"])
src, trg = data_config['text'].split(',')
out = predict_from_run(args.p, run = 0, sentence = data['train'][args.n][src])
print(out[0])
print({'gold_text': data['train'][args.n][trg]})