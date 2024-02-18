import argparse
import hashlib
import json
import os


parser = argparse.ArgumentParser()

# general args
parser.add_argument('--cache_dir', type=str, default='/newdisk/reflective_thinking', help='cache directory')
parser.add_argument('--dataset', type=str, choices=['truthfulqa', 'hotpotqa'], help='name of the dataset')
parser.add_argument('--k', type=int, default=-1, help='index for parallelized processing')

parser.add_argument('--ckpt_dir', type=str, default='llama-2-7b-chat/')
parser.add_argument('--tokenizer_path', type=str, default='tokenizer.model')
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--max_batch_size', type=int, default=6)

# preproces:
parser.add_argument('--save_hidden_states', action='store_true', default=False, help='whether to save model hidden states during preprocessing')
parser.add_argument('--train_from_path', action='store_true', default=False, help='loading a model')
parser.add_argument('--ft_gpt2_path', type=str, default='/share/data/speech/yanhong/retrieval_lm/wandb/retrieval_lm/gpt2_ft_1024/gpt2_ft_3e-4/epoch=10-step=18786.ckpt', help='finetuned gpt2-small')
parser.add_argument('--ft_gpt2_xl_path', type=str, default='/share/data/speech/yanhong/retrieval_lm/wandb/retrieval_lm/gpt2-xl/1e-6/last.ckpt', help='finetuned gpt2-xl')

parser.add_argument('--k_interval', type=int, default=-1, help='chunk iterval for parallelized processing')


# n-gram
parser.add_argument('--ngram_only', action='store_true', default=False, help='use ngram to constuct datastore')
parser.add_argument('--min_ngram_size', type=int, default=2, help='minimum ngram size')
parser.add_argument('--max_ngram_size', type=int, default=20, help='maximum ngram size')
parser.add_argument('--min_occur_count', type=int, default=5, help='minimum occur count')

# use token probs to extract phrases
parser.add_argument('--use_token_probs', action='store_true', default=False, help='use token probabilities to constuct datastore')
parser.add_argument('--tok_prob_threshold', type=float, default=0.5, help='token probability threshold')

parser.add_argument('--similarity_threshold', type=float, default=0.5, help='similarity threshold for retrieval')

parser.add_argument('--make_token_trie', action='store_true', default=False, help='make token trie in parallel')
parser.add_argument('--token_trie_idx', type=int, default=0, help='each job now only makes 1 token trie')

parser.add_argument('--gen_file_name', type=str, default='0.json', help='name for text generation file')
parser.add_argument('--compute_speed', action='store_true', default=False, help='log the inference time')
parser.add_argument('--test_mode', action='store_true', default=False, help='test mode')

parser.add_argument('--accumulate_grad_batches', type=int, default=0, help='use grad accumulation')

parser.add_argument('--lm_baseline_only', action='store_true', default=False, help='only calculate lm baseline in eval.py')



# prepare_MTbench_dataset.py
parser.add_argument('--create_MTbench_val_test', action='store_true', default=False, help='create the val and test set for MTbench')

# used for 'distribution.launch'
parser.add_argument('--use_ddp', action='store_true', default=False, help='use distributed training')
parser.add_argument('--num_gpus', type=int, default=-1, help='specify the number of GPUs, -1 is all available')
parser.add_argument('--local-rank', type=int, default=-1, help='local process rank')

# wandb
parser.add_argument('--use_wandb', action='store_true', default=False, help='log with wandb')
parser.add_argument('--wandb_project', type=str, help='wandb project to log in')
parser.add_argument('--wandb_group', type=str, help='wandb group for runs')
parser.add_argument('--wandb_dir', type=str, help='base wandb directory')
parser.add_argument('--wandb_name', type=str, help='wandb run id')


parser.add_argument('--seed', type=int, default=42)


my_config = parser.parse_args()