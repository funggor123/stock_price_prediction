#!/usr/bin/python3
from datetime import datetime


def cron_news(dbclient, start="2019-01-05-11-23-23", end="2020-05-10-11-23-23", field="Accessories & Leather Goods"):
    start = datetime.strptime(start, '%Y-%m-%d-%H-%M-%S')
    end = datetime.strptime(end, '%Y-%m-%d-%H-%M-%S')

    fyp_db = dbclient["fyp"]
    news_col = fyp_db["news"]

    query = {"$and": [{"date": {"$lt": end, "$gt": start}}, {"field": field}]}
    news = news_col.find(query).sort("date")
    return news


def cron_sentiment(dbclient, news_id_set):
    fyp_db = dbclient["fyp"]
    sentiment_col = fyp_db["sentiment"]

    query = {"newsID": {"$in": news_id_set}}
    sentiments = sentiment_col.find(query)

    sentiment_map = {sentiment['newsID']: sentiment for sentiment in sentiments}
    ordered_sentiments = [sentiment_map[news_id] for news_id in news_id_set if news_id in sentiment_map]

    return ordered_sentiments


def get_sentiment_scores(dbclient, start="2019-01-05-11-23-23", end="2020-05-10-11-23-23",
                         field="Accessories & Leather Goods"):
    sentiments = cron_sentiment(dbclient, [new['_id'] for new in cron_news(dbclient, start, end, field)])
    return [float(sentiment['score']) for sentiment in sentiments]
