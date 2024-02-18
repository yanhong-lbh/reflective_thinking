from typing import List, Optional
import fire
from llama import Llama, Dialog
from datasets import load_dataset
from my_config import my_config
import os
import json

def format_context(context):
    formatted_context = ""
    for title, sentences in zip(context['title'], context['sentences']):
        formatted_context += f"{title}: " + " ".join(sentences) + " "
    return formatted_context

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.8,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    dataset: str=None,
    k : int=1,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    cache_dir = '/newdisk/reflective_thinking'

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if dataset == 'truthfulqa':
        raw_datasets = load_dataset('domenicrosati/TruthfulQA', cache_dir=cache_dir)
        all_questions = raw_datasets['train']['Question']
    elif dataset == 'hotpotqa':
        raw_datasets = load_dataset('hotpot_qa', 'distractor', cache_dir=cache_dir)['train'][:10000]
        all_questions = raw_datasets['question']
        all_contexts = raw_datasets['context']

    for q_idx, question in enumerate(all_questions):

        if q_idx < 20 * k or q_idx >= 20 * (k + 1):
            continue

        init_responses_dir = f'{cache_dir}/{dataset}/init_responses'
        os.makedirs(init_responses_dir, exist_ok=True)

        init_responses_path = f'{init_responses_dir}/{q_idx}.json'

        if not os.path.exists(init_responses_path):

            all_responses = []

            if dataset == 'truthfulqa':
                dialogs: List[Dialog] = [
                    [{"role": "user", "content": f"{question}"}],
                    [{"role": "user", "content": f"{question}"}],
                    [{"role": "user", "content": f"{question}"}],
                    [{"role": "user", "content": f"{question}"}],
                ]
            elif dataset == 'hotpotqa':
                context = all_contexts[q_idx]
                formatted_context = format_context(context)
                if len(formatted_context) >= 8000:
                    print("context too long; disregard")
                    continue

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}
                    ],
                ]

    
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=384,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                all_responses.append(result['generation']['content'])

            with open(init_responses_path, 'w') as f:
                json.dump(all_responses, f)

        init_critiques_dir = f'{cache_dir}/{dataset}/init_critiques'
        os.makedirs(init_critiques_dir, exist_ok=True)

        init_critiques_path = f'{init_critiques_dir}/{q_idx}.json'

        if not os.path.exists(init_critiques_path):

            with open(init_responses_path, 'r') as f:
                all_responses = json.load(f)

            all_critiques = []

            if dataset == 'truthfulqa':

                dialogs: List[Dialog] = [
                    [
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": "Could you critique your last response?"},
                    ],
                    [
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": "Could you critique your last response?"},
                    ],
                    [
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": "Could you critique your last response?"},
                    ],
                    [
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": "Could you critique your last response?"},
                    ],
                ]

            elif dataset == 'hotpotqa':
                context = all_contexts[q_idx]
                formatted_context = format_context(context)
                if len(formatted_context) >= 8000:
                    print("context too long; disregard")
                    continue

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": f"Please review and critique your previous response. You can refer back to the original context if needed."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": f"Please review and critique your previous response. You can refer back to the original context if needed."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": f"Please review and critique your previous response. You can refer back to the original context if needed."}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": f"Please review and critique your previous response. You can refer back to the original context if needed."}
                    ],
                ]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                all_critiques.append(result['generation']['content'])

            with open(init_critiques_path, 'w') as f:
                json.dump(all_critiques, f)


        res_wo_ref_dir = f'{cache_dir}/{dataset}/res_wo_ref'
        os.makedirs(res_wo_ref_dir, exist_ok=True)

        res_wo_ref_path = f'{res_wo_ref_dir}/{q_idx}.json'

        if not os.path.exists(res_wo_ref_path):

            with open(init_responses_path, 'r') as f:
                all_responses = json.load(f)

            if dataset == 'truthfulqa':

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": f"{question}"},
                    ]
                ]

            elif dataset == 'hotpotqa':
                context = all_contexts[q_idx]
                formatted_context = format_context(context)

                if len(formatted_context) >= 8000:
                    print("context too long; disregard")
                    continue

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                    ]
                ]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                res_wo_ref = result['generation']['content']

            with open(res_wo_ref_path, 'w') as f:
                json.dump(res_wo_ref, f)


        res_w_ref_dir = f'{cache_dir}/{dataset}/res_w_ref'
        os.makedirs(res_w_ref_dir, exist_ok=True)

        res_w_ref_path = f'{res_w_ref_dir}/{q_idx}.json'

        if not os.path.exists(res_w_ref_path):

            with open(init_responses_path, 'r') as f:
                all_responses = json.load(f)

            with open(init_critiques_path, 'r') as f:
                all_critiques = json.load(f)

            if dataset == 'truthfulqa':

                if q_idx in [153, 192, 504]:
                    continue

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Provide extremely concise answers with no explanation."},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": "Please review and critique your previous response."},
                        {"role": "assistant","content": f"{all_critiques[0]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": "Please review and critique your previous response."},
                        {"role": "assistant","content": f"{all_critiques[1]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": "Please review and critique your previous response."},
                        {"role": "assistant","content": f"{all_critiques[2]}"},
                        {"role": "user", "content": f"{question}"},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": "Please review and critique your previous response."},
                        {"role": "assistant","content": f"{all_critiques[3]}"},
                        {"role": "user", "content": f"{question}"},
                    ]
                ]

            elif dataset == 'hotpotqa':
                context = all_contexts[q_idx]
                formatted_context = format_context(context)

                if len(formatted_context) >= 8000:
                    print("context too long; disregard")
                    continue

                dialogs: List[Dialog] = [
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[0]}"},
                        {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
                        {"role": "assistant","content": f"{all_critiques[0]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[1]}"},
                        {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
                        {"role": "assistant","content": f"{all_critiques[1]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[2]}"},
                        {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
                        {"role": "assistant","content": f"{all_critiques[2]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                        {"role": "assistant","content": f"{all_responses[3]}"},
                        {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
                        {"role": "assistant","content": f"{all_critiques[3]}"},
                        {"role": "user", "content": f"{question}\n Provide a short answer without explanation."},
                    ]
                ]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for dialog, result in zip(dialogs, results):
                res_w_ref = result['generation']['content']

            with open(res_w_ref_path, 'w') as f:
                json.dump(res_w_ref, f)


if __name__ == "__main__":
    fire.Fire(main)
