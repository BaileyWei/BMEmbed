import os
import json
import datetime
import random
import argparse
import numpy as np
from tqdm import tqdm

from bm25_retrieval import BMScorer
from partitioning_strategy import generate_intervals

def data_sampler(args):
    # set seed
    random.seed(args.seed)

    # read files
    doc2query = json.load(open(f'../data/sythetic_data/{args.dataset}/doc2query.json'))
    chunked_corpus = json.load(open(f'../data/sythetic_data/{args.dataset}/chunked_corpus.json'))
    documents = []
    for data in chunked_corpus:
        documents.append(data['chunked_text'])

    print('bm25 model preparing.....')
    model = BMScorer(documents,
                     k_1=args.bm25_k1,
                     b=args.bm25_b)

    print('bm25 model searching.....')
    for data in tqdm(doc2query):
        topk_list = model.retrieve(data['query'], percentile=args.percentile)
        data['bm25_searching'] = topk_list

    print('bm25 dataset making.....')
    train_dataset = []
    for example in tqdm(doc2query):
        # divide negative intervals while keep positive sample
        positive_interval = [0, args.first_sample_rank_range]
        negative_intervals = generate_intervals(start_range=example['bm25_searching'][-1]['score'],
                                                end_range=example['bm25_searching'][args.first_sample_rank_range]['score'],
                                                m=args.m - 1,
                                                interval_multiplier=args.interval_multiplier,
                                                score=args.score,
                                                num_samples=args.num_samples)
        positives = random.sample(example['bm25_searching'][positive_interval[0]:positive_interval[1]], args.num_samples)
        scores = [1/(float(item['score']) if float(item['score']) else 1e-5) for item in example['bm25_searching'][args.first_sample_rank_range:]]
        negatives = []
        last_end_index = None
        for start, end in negative_intervals[::-1]:
            # if last round has extended the interval, in this round we exclude these samples
            if last_end_index:
                start_index = last_end_index
                last_end_index = None
            else:
                start_index = np.searchsorted(scores, 1/(end if end else 1e-5), side='left')

            end_index = np.searchsorted(scores, 1/(start if start else 1e-5), side='right')
            # if no samples in this interval, extend to next interval for few samples
            if end_index < start_index + args.num_samples:
                end_index = start_index + args.num_samples
                last_end_index = end_index

            interval_samples = example['bm25_searching'][args.first_sample_rank_range:][start_index:end_index]
            # print(start_index,end_index)
            num_negative = random.sample(interval_samples, args.num_samples)
            negatives.append(num_negative)

        for i in range(len(positives)):
            positive = positives[i]

            negative_candidates = [negative[i] for negative in negatives]
            train_dataset.append({'query': example['query'],
                                  'positive': positive,
                                  'negative': negative_candidates})
            # print([positive['score']] + [i['score'] for i in negative_candidates])


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    dataset_path = f'../data/sythetic_data/{args.dataset}/{timestamp}_score/'
    os.makedirs(dataset_path, exist_ok=True)
    print(f'bm25 dataset volume {len(train_dataset)}')

    print(f'bm25 dataset saving as {dataset_path}/bm25_dataset.json')
    with open(f'{dataset_path}/bm25_dataset.json', 'w') as file:
        json.dump(train_dataset, file)


    print(f'bm25 dataset sampling config saving as {dataset_path}/sampling_config.json')
    with open(f'{dataset_path}/sampling_config.json', "w") as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random sampling to ensure reproducibility')
    parser.add_argument('--dataset', type=str, default='multihop-rag',
                        help='Name of the dataset to be used')
    parser.add_argument('--percentile', type=float, default=0.1,
                        help='The percentile of candidates to consider for retrieval')
    parser.add_argument('--m', type=int, default=10,
                        help='Number of intervals to generate in the range')
    parser.add_argument('--interval_multiplier', type=float, default=1,
                        help='Multiplier for each interval size relative to the previous one, \
                        interval_multiplier = 1 means a uniform division')
    parser.add_argument('--first_sample_rank_range', type=int, default=3,
                        help='Rank range for selecting the first sample, default is 0 to 3')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of times to sample each query')
    parser.add_argument('--bm25_k1', type=float, default=1.2,
                        help='BM25 parameter k1, controlling term frequency scaling')
    parser.add_argument('--bm25_b', type=float, default=0.75,
                        help='BM25 parameter b, controlling document length normalization')
    parser.add_argument('--score', type=bool, default=True,
                        help='whether the sampling intervals are divided by bm25 score or not (rank)')


    args = parser.parse_args()
    data_sampler(args)