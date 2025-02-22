import torch
import argparse

import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig

import os
import re
import json
import pickle
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, PretrainedConfig

from retrieval_evaluating import calculate_metrics, calculate_recall

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_sentence_embeds(batch_texts, max_length=1024,model_name=None):
    if model_name == 'qwen':
        batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def mask_query(query, keywords):
    keywords_list = [k.strip() for k in keywords.split(', ')]
    for k in keywords_list:
        if re.search(k.lower(), query.lower()):
            start = re.search(k.lower(), query.lower()).span()[0]
            end = re.search(k.lower(), query.lower()).span()[1]
            query = query[0:start] + '[MASK]' + query[end:]

    return query


def get_database_embeds(template_dataset, device, batch_size=2, temp_emb_path=None,model_name=None):
    num = len(template_dataset)//batch_size

    if not temp_emb_path or not os.path.exists(temp_emb_path):
        template_embeddings = list()
        for i in tqdm(range(num)):
            records = template_dataset[i*batch_size:(i+1)*batch_size]
            template_embeddings.append(get_sentence_embeds(records,model_name=model_name))

        template_embeddings_mat = torch.concat(template_embeddings)
        if temp_emb_path:
            pickle.dump(template_embeddings_mat, open(temp_emb_path, 'wb'))
    else:
        template_embeddings_mat = pickle.load(open(temp_emb_path, 'rb'))
        template_embeddings_mat = template_embeddings_mat.to(device)

    return template_embeddings_mat


def find_top_k(query_text, database_embedding, k=20, model_name=None):
    query_embed = get_sentence_embeds([query_text], model_name=model_name)
    scores = (query_embed @ database_embedding.T) * 100
    rank_idx = torch.sort(scores)[1][0][-k:].tolist()[::-1]
    rank_score = torch.sort(scores)[0][0][-k:].tolist()[::-1]

    top_k_documents = []
    for idx,score in zip(rank_idx,rank_score):
        top_k_documents.append((database[idx], database_id[idx]))

    return top_k_documents


def eval(data):
    retrieved_lists = []
    gold_lists = []

    for d in data:
        if d.get('retrieval', None):
            retrieved_lists.append([m[0] if isinstance(m, list) or isinstance(m, tuple) else m for m in d['retrieval']])
            gold_lists.append(d['evidence_list'])
    # Calculate metrics
    metrics = calculate_metrics(retrieved_lists, gold_lists)

    # Print the metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print('-' * 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='multihop-rag',
                        help='Name of the dataset to be used')
    parser.add_argument('--dataset_file_path', type=str,
                        help='Training data file used by the embedding model')
    parser.add_argument('--base_model', type=str,
                        help='base model dir')
    parser.add_argument('--output_dir', type=str,
                        help='fine-tuned embedding model saved dir')
    parser.add_argument('--listwise', action='store_true',
                        help='training method')
    parser.add_argument('--label_scaling', type=float,
                       )
    parser.add_argument('--save_embedding_database', action='store_true',
                        help='whether save the embedding database')
    parser.add_argument('--save_retrieval_result', action='store_true',
                        help='whether save the retrieval_result')


    args = parser.parse_args()

    method = 'listwise' if args.listwise else 'random'
    name_data_file = args.dataset_file_path.strip('/').split('/')[-1]
    model_path = f'{args.output_dir}/{args.dataset_name}/{method}/{name_data_file}_{args.label_scaling}'

    tokenizer = AutoTokenizer.from_pretrained(
         args.base_model,
         trust_remote_code=True,
         torch_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if not tokenizer.pad_token:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(
        model,
        model_path,

    )
    model = model.merge_and_unload()
    device = 'cuda:1'
    model.to(device)

    # read database
    database = []
    database_id = []

    corpus = json.load(open(f'./data/sythetic_data/{args.dataset_name}/chunked_corpus.json'))
    for d in corpus:
        database.append(d['chunked_text'])
        database_id.append(d['title'] if d.get('title', None) else d.get('id', None))

    if args.save_embedding_database:
        embedding_path = f"{model_path}/embedding_database.pkl"
    else:
        embedding_path=None

    database_embedding = get_database_embeds(database,
                                             device,
                                             temp_emb_path=embedding_path,
                                             model_name = 'qwen'
                                             )
    task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    retrieval_dataset = json.load(open(f'./data/evaluation_set/{args.dataset_name}/retrieval_dataset.json'))

    for data in tqdm(retrieval_dataset):
        query_text = get_detailed_instruct(task_instruction,data['query'])
        top_k_documents = find_top_k(query_text, database_embedding, model_name='qwen',k=20)
        data['retrieval'] = top_k_documents

    eval(retrieval_dataset)

    if args.save_retrieval_result:
        with open(f"{model_path}/retrieval_result.json", 'w') as f:
            json.dump(retrieval_dataset, f)

