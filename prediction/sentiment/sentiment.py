from prediction.sentiment.cron import get_sentiment_scores
import pymongo
import pandas as pd
import math
from prediction.cron.list_xls_csv import get_industries_list

myclient = pymongo.MongoClient("mongodb://34.87.175.206:27017/",
                               username='hkust_fyp_dl1',
                               password='dl1@UST_master')


def get_sentiment_score(dbclient, start="2019-01-05-11-23-23", end="2020-05-10-11-23-23",
                        field="Accessories & Leather Goods"):
    scores = get_sentiment_scores(dbclient, start, end, field)
    df = pd.DataFrame(data=scores, columns=['score'])
    return df['score'].mean(), len(scores)


def get_mass_sentiment_scores(dbclient, start="2019-01-05-11-23-23", end="2020-05-10-11-23-23"):
    print("start to get field's sentiment score")
    industries = get_industries_list()
    industries_score = []
    total_sample_size = 0
    for industry in industries:
        score, sample_size = get_sentiment_score(dbclient, start, end, industry)
        if sample_size == 0:
            score = -2
        total_sample_size += sample_size
        industries_score.append({'score': score, 'field': industry, 'sample_size': sample_size})
    sentiment_summary(industries_score)
    print("finish")
    print("total_size: ", total_sample_size)
    return industries_score


def sentiment_summary(industries_score):
    for score in industries_score:
        print(score)


def get_recommended_fields(dbclient, only_positive=False, top=5, start="2019-12-20-00-00-00",
                           end="2020-01-07-23-59-59"):
    scores = get_mass_sentiment_scores(dbclient, start, end)
    remove_nan_scores = [score for score in scores if not math.isnan(score['score'])]
    top_scores = sorted(remove_nan_scores, key=lambda remove_nan_scores: remove_nan_scores['score'], reverse=True)[:top]
    if only_positive:
        top_scores = [score for score in top_scores if score['score'] < 0]
    return top_scores, scores

'''
top_scores, scores = get_recommended_fields(myclient)
print(top_scores)
'''
