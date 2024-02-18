from datasets import load_dataset
from my_config import my_config
import os
import json
import glob


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

dataset = my_config.dataset
cache_dir = my_config.cache_dir


if dataset == 'truthfulqa':
    raw_datasets = load_dataset('domenicrosati/TruthfulQA', cache_dir=cache_dir)
    all_questions = raw_datasets['train']['Question']
elif dataset == 'hotpotqa':
    raw_datasets = load_dataset('hotpot_qa', 'distractor', cache_dir=cache_dir)['train'][:10000]
    all_questions = raw_datasets['question']
    all_contexts = raw_datasets['context']
    all_answer = raw_datasets['answer']
    all_type = raw_datasets['type']
    all_level = raw_datasets['level']


init_responses_dir = f'{cache_dir}/{dataset}/init_responses'
res_wo_ref_dir = f'{cache_dir}/{dataset}/res_wo_ref'
res_w_ref_dir = f'{cache_dir}/{dataset}/res_w_ref'
files = glob.glob(f"{res_w_ref_dir}/*.json")

q_idx_list = [int(os.path.basename(path).split('.')[0]) for path in files]
q_idx_list.sort()


all_data = {}
for q_idx in q_idx_list:
    init_responses_path = f'{init_responses_dir}/{q_idx}.json'
    res_wo_ref_path = f'{res_wo_ref_dir}/{q_idx}.json'
    res_w_ref_path = f'{res_w_ref_dir}/{q_idx}.json'
    init_responses = read_json(init_responses_path)
    res_wo_ref = read_json(res_wo_ref_path)
    res_w_ref = read_json(res_w_ref_path)
    if dataset == 'truthfulqa':
        all_data[q_idx] = {'Question': all_questions[q_idx], 'init_responses': init_responses, 'res_wo_ref':res_wo_ref, 'res_w_ref': res_w_ref}
    elif dataset == 'hotpotqa':
        question = all_questions[q_idx]
        context = all_contexts[q_idx]
        answer = all_answer[q_idx]
        type = all_type[q_idx]
        level = all_level[q_idx]
        all_data[q_idx] = {'question': question, 'context': context, 'answer': answer, 'type': type, 'level': level, 'init_responses': init_responses, 'res_wo_ref':res_wo_ref, 'res_w_ref': res_w_ref}

save_path = f'all_data_{dataset}.json'

with open(save_path, 'w') as f:
    json.dump(all_data, f)



        

breakpoint()


