import re


def postprocess_events(text, parsed_columns):
    event_extraction_pattern = [(r'(\*\*Event\*\*:\s*.*?(?=\d+\.\s*\*\*Event\*\*:|\Z|\n\n))', '\*\*{column}\*\*:\s*(.*?)\s*(?=\*\*|\Z)'),
                                (r'(\*\*Event:\*\*\s*.*?(?=\d+\.\s*\*\*Event:\*\*|\Z|\n\n))', '\*\*{column}:\*\*\s*(.*?)\s*(?=\-\s|\Z)\-\s'),
                                (r'(\[Event\]:\s*.*?(?=\d+\.\s*\[Event\]:|\Z|\n\n))', '\[{column}\]:\s*(.*?)\s*(?=\[|\Z)')
                                ]

    for event_extraction_pattern_,p  in event_extraction_pattern:
        event_list = re.findall(event_extraction_pattern_, text, re.DOTALL)
        if event_list:
            pattern = p
            parsed_columns = ['Event', 'Topic', 'Original context', 'Type']
            columns_extraction_pattern = ''
            for c in parsed_columns[:-1]:
                columns_extraction_pattern += pattern.format(column=c)
            break


    parsed_dicts = []
    for event in event_list:
        e_dict = {}
        results = re.findall(columns_extraction_pattern, event, re.DOTALL)
        if results:
            for key, value in zip(parsed_columns, list(results[0])):
                if key == 'Original context':
                    _ = re.findall(r'"(.*?)"', value, re.DOTALL)
                    if not re.findall(r'"(.*?)"', value, re.DOTALL):
                        value = [value]
                    else:
                        value = re.findall(r'"(.*?)"', value, re.DOTALL)

                e_dict[key] = value
            e_dict['plain_text'] = event
            parsed_dicts.append(e_dict)
    return parsed_dicts


def postprocess_queries(text):
    query_list = []
    pattern = [r'\[Event\]:\s*(.*?)\s*\[Question\]:\s*(.*?)\s*\[Original context\]:\s*(.*?)(?=\n\n|\Z)',
               r'\*\*Event\*\*:\s*(.*?)\s*\*\*Question\*\*:\s*(.*?)\s*\*\*Original context\*\*:\s*(.*?)(?=\n\n|\Z)',
               r'\*\*Event:\*\*\s*(.*?)\s*\*\*Question:\*\*\s*(.*?)\s*\*\*Original Reference Paragraph:\*\*\s*(.*?)(?=\n\n|\Z)',
               r'\[Event\]:\s*(.*?)\s*\[Question\]:\s*(.*?)\s*\[Original reference paragraph\]:\s*(.*?)(?=\n\n|\Z)',
               ]
    for p in pattern:
        matches = re.findall(p, text, re.DOTALL)
        if matches:
            break
    #     original_context = re.findall(r'"(.*?)"', original_context,re.DOTALL)
    for match in matches:
        event, question, original_context = match
        original_context_list = re.findall(r'"(.*?)"', original_context, re.DOTALL)
        if not original_context_list:
            original_context_list = [original_context]
        query_list.append({
            'event': event,
            'question': question,
            'original_context': original_context_list
        })
    return query_list


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


def find_evidence(text, document):
    pattern = r'["“](.*?)["”]'
    replaced_document = re.sub(pattern, r"'\1'", document)
    replaced_c = re.sub(pattern, r"'\1'", text)
    replaced_c = replaced_c.replace(' ,', ',')
    document_sentences = sent_tokenize(document)
    text_sentences = sent_tokenize(text)
    evidence_list = []
    for text_sentence in text_sentences:
        for document_sentence in document_sentences:
            pattern = r'["“](.*?)["”]'
            replaced_document = re.sub(pattern, r"'\1'", document_sentence)
            replaced_text = re.sub(pattern, r"'\1'", text_sentence)
            replaced_text = replaced_text.replace(' ,', ',')

            text_list = word_tokenize(replaced_text)
            sentence_list = word_tokenize(replaced_document)
            #             print(text_list)
            #             print(sentence_list)
            edit_distance = minDistance(text_list, sentence_list)
            #             print(edit_distance, len(sentence_list), len(text_list))
            if abs(len(text_list) - edit_distance) / len(sentence_list) > 0.85 or \
                    abs(len(sentence_list) - edit_distance) / len(text_list) > 0.85:
                evidence_list.append(document_sentence)
                break
    return ' '.join(evidence_list)


def minDistance(list1, list2) -> int:
    m = len(list1) + 1
    n = len(list2) + 1
    # 1,2 min distance between word1[:1] word2[:2]
    dp_table = [[0] * n for _ in range(m)]
    for i in range(1, m):
        dp_table[i][0] = i
    for j in range(1, n):
        dp_table[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if list1[i - 1] == list2[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1]

            else:
                # i-1 = j
                # i = j-1
                # i-1 = j-1
                dp_table[i][j] = min(dp_table[i - 1][j] + 1,
                                     dp_table[i - 1][j - 1] + 1,
                                     dp_table[i][j - 1] + 1)
    return dp_table[-1][-1]