def calculate_metrics(retrieved_lists, gold_lists):
    hits_at_10_count = 0
    hits_at_4_count = 0
    hits_at_1_count = 0

    recall = 0

    map_at_10_list = []
    mrr_list = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        hits_at_10_flag = False
        hits_at_4_flag = False
        hits_at_1_flag = False
        average_precision_sum = 0
        first_relevant_rank = None
        find_gold = []

        gold = [item.replace(" ", "").replace("\n", "") for item in gold]
        retrieved = [item.replace(" ", "").replace("\n", "") for item in retrieved]

        # retrieved = [item[0].replace(" ", "").replace("\n", "") for item in retrieved]

        for rank, retrieved_item in enumerate(retrieved[:41], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 4:
                        hits_at_4_flag = True
                    if rank <= 1:
                        hits_at_1_flag = True
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count = count + 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank

        recall += calculate_recall(gold, retrieved)
        # Calculate metrics for this query
        hits_at_10_count += int(hits_at_10_flag)
        hits_at_4_count += int(hits_at_4_flag)
        hits_at_1_count += int(hits_at_1_flag)
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)

    # Calculate average metrics over all queries
    hits_at_10 = hits_at_10_count / len(gold_lists)
    hits_at_4 = hits_at_4_count / len(gold_lists)
    hits_at_1 = hits_at_1_count / len(gold_lists)

    map_at_10 = sum(map_at_10_list) / len(gold_lists)
    mrr_at_10 = sum(mrr_list) / len(gold_lists)
    recall = recall / len(gold_lists)

    return {
        'num': len(gold_lists),
        'Hits@10': hits_at_10,
        'Hits@4': hits_at_4,
        'Hits@1': hits_at_1,
        'MAP@10': map_at_10,
        'MRR@10': mrr_at_10,
        'recall@20': recall
    }


def calculate_recall(gold_list, retrieved_list, topk=20):
    hit = 0
    d = {gold: 0 for gold in gold_list}
    for retrieved_item in retrieved_list[:topk]:
        for gold in gold_list:
            if gold in retrieved_item:
                d[gold] += 1
                hit += 1
                break
    # print(hit)
    return 1 - sum(1 for key in d if d[key] == 0) / len(gold_list)
