import os
import json
import nltk
import argparse
nltk.download('punkt_tab')

from tqdm import tqdm
from typing import Any, Generator, List, Optional
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import RegexpTokenizer

def chunk_text(text, tokenizer=None, max_length=256, overlap_sentence_length=1, ):
    # split sentences
    re_tokenizer = RegexpTokenizer(r'[^@\n!?;]+[@\n!?;]')
    text = text.replace('. ', '@')
    if text[-1] == '.':
        text = text + '@'
    sentences = re_tokenizer.tokenize(text)


    chunks = []
    current_chunk = []
    current_chunk_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer(sentence)['input_ids'])

        # add sentences to current chunk
        if current_chunk_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_chunk_length += sentence_length
        else:
            # add to chunks
            chunks.append(' '.join(current_chunk).replace('@', '.'))

            # remain overlap
            overlap_chunk = ' '.join(current_chunk[-overlap_sentence_length:])
            current_chunk = [overlap_chunk, sentence]
            current_chunk_length = len(overlap_chunk) + sentence_length

    # last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).replace('@', '.'))

    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        help='base model dir')
    parser.add_argument('--corpus_file_path', type=str)
    parser.add_argument('--dataset_name', default='multihop-rag', type=str)
    args = parser.parse_args()

    nltk.download('punkt_tab')

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)


    corpus = json.load(open(args.corpus_file_path, 'r'))

    n = 0
    chunked_corpus = []
    for doc in tqdm(corpus):
        chunked_list = chunk_text(doc['text'], tokenizer=tokenizer, max_length=256, overlap_sentence_length=1)
        for chunk in chunked_list:
            chunked_corpus.append(
                {'chunked_text': chunk,
                 'title': doc['title']
                 }
            )
            n +=1

    with open(f'../data/sythetic_data/{args.dataset_name}/chunked_corpus.json', 'w') as f:
            json.dump(chunked_corpus, f)
    print(n)


