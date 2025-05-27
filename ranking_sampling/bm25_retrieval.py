import string
import nltk
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


nltk.download('punkt')

class BMScorer:
    def __init__(self, documents, k_1=1.2, b=0.75):
        self.k_1 = k_1
        self.b = b
        self.word_tokenzier = word_tokenize
        self.documents = documents
        tokenized_documents = self.tokenize(documents)
        self.model = BM25Okapi(tokenized_documents, k1=k_1, b=b)

    def tokenize(self, text):
        if isinstance(text, str):
            return [word for word in self.word_tokenzier(text.lower()) if word not in string.punctuation]
        elif isinstance(text, list):
            return [[w for w in self.word_tokenzier(doc.lower()) if w not in string.punctuation] for doc in text]
        else:
            raise TypeError('Input must be a list of strings or strings')

    def retrieve(self, query, topk=None, percentile=None):
        top_list = []
        tokenized_query = self.tokenize(query)
        scores = self.model.get_scores(tokenized_query)
        sorted_list_with_index = sorted(enumerate(scores), key=lambda x: x[1])
        if topk and not percentile:
            for i in sorted_list_with_index[::-1][:topk]:
                top_list.append(
                    {'index': i[0],
                     'score': i[1],
                     'text': self.documents[i[0]]
                     }
                )
        elif percentile and not topk:
            topk = int(percentile * len(self.documents))
            for i in sorted_list_with_index[::-1][:topk]:
                top_list.append(
                    {'index': i[0],
                     'score': i[1],
                     'text': self.documents[i[0]]
                     }
                )
        else:
            raise TypeError('Please define either topk or percentile to retrieval documents')

        return top_list



if __name__ == "__main__":
    doc2query = json.load(open('../data/sythetic_data/multihop-rag/doc2query.json'))
    chunked_corpus = json.load(open('../data/sythetic_data/multihop-rag/chunked_corpus.json'))
    documents = []
    for data in chunked_corpus:
        documents.append(data['chunked_text'])
    print('bm25 model preparing.....')
    model = BMScorer(documents)


    for data in tqdm(doc2query):
        topk_list = model.retrieve(data['query'],topk=10)
        data['bm25_searching'] = topk_list



