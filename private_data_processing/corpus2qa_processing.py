import os
import json
import openai
import argparse
from tqdm import tqdm
from utils import postprocess_events, postprocess_queries, find_evidence
from prompts import doc2event_prompt, event2qa_prompt

def get_events(content, title=None):
    if title:
        doc = f"Title:{title}\n\n{content}"
    else:
        doc = f"{content}"

    try:
        generated_events = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": doc2event_prompt.format(doc=doc)}
            ]
        )
        gpt4_generated_events = generated_events.choices[0].message['content']
    except:
        gpt4_generated_events = ''
    return gpt4_generated_events

def get_queries(content, event_list):
    event = ''
    doc = content
    for i, e in enumerate(event_list):
        event += f"{i + 1}.\n{e['plain_text']}\n\n"

    try:
        generated_queries = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": event2qa_prompt.format(doc=doc,event=event)}
            ]
        )
        gpt4_generated_queries = generated_queries.choices[0].message['content']
    except:
        gpt4_generated_queries = ''

    return gpt4_generated_queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        help='base model dir')
    parser.add_argument('--corpus_file_path', type=str)
    parser.add_argument('--dataset_name', default='multihop-rag', type=str)
    args = parser.parse_args()

    doc2qa = 'sk-xx'
    openai.api_key = doc2qa
    os.makedirs(f'../data/sythetic_data/{args.dataset_name}/', exist_ok=True)
    corpus = json.load(open(args.corpus_file_path))
    if not os.path.exists(f'../data/sythetic_data/{args.dataset_name}/doc2event.json'):
        for example in tqdm(corpus):
            gpt4_generated_events = get_events(
                                                content=example['body'],
                                                title=example['title']
                                            )
            example['gpt4_generated_events'] = gpt4_generated_events
        with open(f'../data/sythetic_data/{args.dataset_name}/doc2event.json', 'w') as f:
            json.dump(corpus, f)
    else:
        corpus = json.load(open(f'../data/sythetic_data/{args.dataset_name}/doc2event.json'))

    if not os.path.exists(f'./data/sythetic_data/{args.dataset_name}/event2query.json'):
        for example in tqdm(corpus):
            parsed_columns = ['Event', 'Topic', 'Original context', 'Type']
            example['event_list'] = postprocess_events(example['gpt4_generated_events'], parsed_columns)
            gpt4_generated_queries = get_queries(
                                                    content=example['body'],
                                                    event_list=example['event_list']
                                                )
            example['gpt4_generated_queries'] = gpt4_generated_queries

        # save
        with open(f'../data/sythetic_data/{args.dataset_name}/event2query.json', 'w') as f:
            json.dump(corpus, f)
    else:
        corpus = json.load(open(f'../data/sythetic_data/{args.dataset_name}/event2query.json'))


    qa_database = []
    for example in tqdm(corpus):
        if example['event_list']:
            example['query_list'] = postprocess_queries(example['gpt4_generated_queries'])
            events_dict = {e['Event']: e['Original context'] for e in example['event_list']}
            for dic in example['query_list']:
                evidences = []
                if events_dict.get(dic['event'], ''):
                    query = dic['question']
                    for c in events_dict[dic['event']]:
                        evidence_text = find_evidence(c, example['body'])
                        if evidence_text:
                            evidences.append(evidence_text)

                    qa_database.append(
                        {'query': query,
                         'evidence_list': evidences,
                         'doc': example['title'],
                         'event': dic['event']
                         }
                    )

    # save
    with open(f'../data/sythetic_data/{args.dataset_name}/doc2query.json', 'w') as file:
        json.dump(qa_database, file)













