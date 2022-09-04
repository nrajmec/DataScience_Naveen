import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("Known positives:")

        for x in known_positives[:5]:
            print("%s" % x)

        print("Recommended:")

        for x in top_items[:5]:
            print("%s" % x)

sample_recommendation(model, data, [68, 25, 450])