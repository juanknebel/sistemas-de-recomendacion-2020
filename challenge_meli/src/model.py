import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import math
from random import sample, seed, randint, shuffle
import csv


SEED = 42


def dcg(predicted, effective_item):
    dcg = 0
    # map(lambda p, e: relevance(p, e, items), predicted, effective)
    for pos, (item, domain) in enumerate(predicted.items()):
        dcg += relevance((item, domain), effective_item) / (math.log10(2 + pos))
    return dcg


def relevance(predicted_item, effective_item):
    if predicted_item[0] == effective_item[0]:
        return 12
    elif predicted_item[1] == effective_item[1]:
        return 1
    else:
        return 0


def idcg(number_of_items=10):
    # 12/math.log10(1 + 1) + 1/math.log10(1 + 2) + 1/math.log10(1 + 3) ....
    first = 12 / math.log10(1 + 1)
    all_except_first = sum(
        map(
            lambda position: 1 / math.log10(1 + position),
            range(2, number_of_items + 1),
        )
    )
    return first + all_except_first


def ndcg(predicted, effective_item, items):
    predicted_dict = {}
    for one_item in predicted:
        predicted_dict[one_item] = items[
            items.item_id == one_item
        ].domain_id.iloc[0]
    effective_item = (
        effective_item,
        items[items.item_id == effective_item].domain_id.iloc[0],
    )
    return dcg(predicted_dict, effective_item) / idcg()


def split_train_test_ids(list_of_ids, percentage=70):
    # np.random.choice(a, 3, replace=False)
    until = int(len(list_of_ids) * (percentage / 100))
    seed(SEED)
    shuffle(list_of_ids)
    return list_of_ids[:until], list_of_ids[until:]


def generate_dictionary_by_session(session_history):
    # genero un diccionario por session con otro diccionario que tiene el id del item visto y la cantidad de veces que lo vio. Ordenado desde el último visto como primero del diccionario
    # {session_id: {item_id: how_many_times_saw}, ....}
    recommended_by_session = {}
    for id, group in tqdm(session_history.groupby("session_id")):
        last_items_view = group[group.event_type == "view"].event_info.tolist()
        last_items_view.reverse()
        last_items_view = list(map(lambda item: int(item), last_items_view))
        last_items_view_quantity = {}
        for an_item in last_items_view:
            if an_item in last_items_view_quantity:
                last_items_view_quantity[an_item] += 1
            else:
                last_items_view_quantity[an_item] = 1

        recommended_by_session[id] = last_items_view_quantity

    return recommended_by_session


def generate_dictionary_by_domain(items):
    items_by_domain = {}
    for name, group in items.groupby("domain_id"):
        items_by_domain[name] = group.item_id.tolist()

    return items_by_domain


def generate_submission(filaname, predictions):
    with open(filaname, "w") as file:
        write = csv.writer(file)
        write.writerows(predictions)


def predict(recommendations, random_item, items, items_by_domain):
    predictions = []
    for _, recommendations in tqdm(recommendations.items()):
        # first 10 to predict
        predicted = list(recommendations.keys())[:10]

        if len(predicted) < 10:
            # sort items by views
            try:
                most_viewed = list(
                    {
                        k: v
                        for k, v in sorted(
                            recommendations.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    }
                )[0]
            except:
                most_viewed = random_item

            most_viewed_by_domain = items[
                items.index == most_viewed
            ].domain_id.iloc[0]

            predicted += items_by_domain[most_viewed_by_domain][
                : 10 - len(predicted)
            ]
            for i in range(10 - len(predicted)):
                seed(i)
                onther_random_item = items.loc[
                    randint(0, len(items) - 1)
                ].item_id
                predicted.append(onther_random_item)

        predictions.append(predicted)

    return predictions


def generate_random_item(items):
    seed(SEED)
    return items.loc[randint(0, len(items) - 1)].item_id


seed(SEED)
dtype = {
    "item_id": "int",
    "title": "str",
    "domain_id": "str",
    "price": "float",
    "category_id": "str",
    "condition": "str",
    "site": "str",
}
df_items = pd.read_csv("../data/02_intermediate/item_data.csv", dtype=dtype)
df_items.set_index("item_id")
df_train = pd.read_csv("../data/03_primary/test_dataset.csv")

recommendations = generate_dictionary_by_session(df_train)
items_by_domain = generate_dictionary_by_domain(df_items)

random_item = generate_random_item(df_items)
predictions = predict(recommendations, random_item, df_items, items_by_domain)

generate_submission(
    "../data/07_model_output/submission_baseline_1.csv", predictions
)
