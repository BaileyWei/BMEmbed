import os
import json
import datetime
import random
import argparse
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
    if args.num_queries > 0:
        doc2query = random.sample(doc2query, args.num_queries)


    for data in chunked_corpus:
        documents.append(data['chunked_text'])

    print('bm25 model preparing.....')
    model = BMScorer(documents,
                     k_1=args.bm25_k1,
                     b=args.bm25_b)

    print('bm25 model searching.....')
    for data in tqdm(doc2query):
        topk_list = model.retrieve(data['query'], topk=args.topk)
        data['bm25_searching'] = topk_list

    if args.strategy == 'fine-to-coarse':
        positive_interval = [0, args.first_sample_rank_range]
        negative_intervals = generate_intervals(start_range=args.first_sample_rank_range,
                                                end_range=args.topk,
                                                m=args.m-1,
                                                interval_multiplier=args.interval_multiplier,
                                                num_samples=args.num_samples)
    elif args.strategy == 'uniform':
        positive_interval = [0, args.first_sample_rank_range]
        negative_intervals = generate_intervals(start_range=args.first_sample_rank_range, end_range=args.topk, m=args.m-1,
                                                interval_multiplier=1,
                                                num_samples=args.num_samples)
    else:
        raise TypeError('stategy should be either fine-to-coarse or uniform')

    args.intervals = positive_interval + negative_intervals

    print(f'start:{positive_interval[0]} end:{positive_interval[1]}')
    for interval in negative_intervals:
        print(f'start:{interval[0]} end:{interval[1]}')

    print('bm25 dataset making.....')
    train_dataset = []
    for example in tqdm(doc2query):
        pairs = []
        num_positive = random.sample([i for i in range(positive_interval[0], positive_interval[1])], args.num_samples)
        negatives = []
        for interval_negative in negative_intervals:
            #         print(interval_positive,interval_negative)
            num_negative = random.sample([i for i in range(interval_negative[0], interval_negative[1])],
                                         args.num_samples)
            negatives.append(num_negative)

        for i in range(len(num_positive)):
            positive = {
                'index': example['bm25_searching'][num_positive[i]]['index'],
                'score': example['bm25_searching'][num_positive[i]]['score'],
                'text': example['bm25_searching'][num_positive[i]]['text']
            }

            negative = []

            for num in negatives:
                negative.append({
                    'index': example['bm25_searching'][num[i]]['index'],
                    'score': example['bm25_searching'][num[i]]['score'],
                    'text': example['bm25_searching'][num[i]]['text']
                }
                )

            train_dataset.append({'query': example['query'],
                                  'positive': positive,
                                  'negative': negative})


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    dataset_path = f'../data/sythetic_data/{args.dataset}/{timestamp}/'
    os.makedirs(dataset_path, exist_ok=True)
    print(f'bm25 dataset volume {len(train_dataset)}')

    print(f'bm25 dataset saving as {dataset_path}/bm25_dataset.json')
    with open(f'{dataset_path}/bm25_dataset.json', 'w') as file:
        json.dump(train_dataset, file)


    print(f'bm25 dataset sampling config saving as {dataset_path}/sampling_config.json')
    with open(f'{dataset_path}/sampling_config.json', "w") as f:
        json.dump(vars(args), f, indent=4)

    # return f'{dataset_path}/bm25_dataset.json'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random sampling to ensure reproducibility')
    parser.add_argument('--dataset', type=str, default='multihop-rag',
                        help='Name of the dataset to be used')
    parser.add_argument('--topk', type=int, default=1000,
                        help='The top-k number of candidates to consider for retrieval')
    parser.add_argument('--m', type=int, default=10,
                        help='Number of intervals to generate in the range')
    parser.add_argument('--strategy', type=str, default='fine-to-coarse',
                        help='Strategy for generating intervals; options include fine-to-coarse or uniform')
    parser.add_argument('--interval_multiplier', type=float, default=2,
                        help='Multiplier for each interval size relative to the previous one')
    parser.add_argument('--first_sample_rank_range', type=int, default=3,
                        help='Rank range for selecting the first sample, default is 0 to 3')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of times to sample each query')
    parser.add_argument('--num_queries', type=int, default=-1,
                        help='sythetic queries volume of the corpus')
    parser.add_argument('--bm25_k1', type=float, default=1.2,
                        help='BM25 parameter k1, controlling term frequency scaling')
    parser.add_argument('--bm25_b', type=float, default=0.75,
                        help='BM25 parameter b, controlling document length normalization')


    args = parser.parse_args()
    data_sampler(args)