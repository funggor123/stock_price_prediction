from prediction.sentiment.sentiment import get_recommended_fields
from prediction.cron.list_xls_csv import get_stocks_code_from_stock_list
from prediction.train.predict import get_predict_test_set
import pymongo
from datetime import datetime, timedelta

myclient = pymongo.MongoClient("mongodb://34.87.175.206:27017/",
                               username='hkust_fyp_dl1',
                               password='dl1@UST_master')
fyp_db = myclient["fyp"]
predict_col = fyp_db["predict"]


def get_recommendations(date):
    sentiment_date = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S') - timedelta(days=7)
    only_date = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').date()
    # First Filters
    top_scores, scores = get_recommended_fields(myclient, top=5, start=sentiment_date.strftime('%Y-%m-%d-%H-%M-%S'),
                                                end=date)

    for score in top_scores:
        stock_rise = []
        stocks_code = get_stocks_code_from_stock_list(score['field'])
        # Prediction
        for stock_code in stocks_code:
            profit = get_predict_test_set(stock_code=stock_code, date=only_date.strftime('%Y-%m-%d'))
            if profit is not None:
                if profit > 0:
                    stock_rise.append({"stock_code": stock_code, "rise": profit})
        score['top_stocks'] = stock_rise

    result_obj = {
        "top": top_scores,
        "date": datetime.strptime(date, '%Y-%m-%d-%H-%M-%S'),
        "other_scores": scores
    }
    return result_obj


START_DATE_TIME_IN_HK = "2020-01-13-09-00-00"
END_DATE_TIME_IN_HK = "2020-03-31-16-00-00"


def get_recommendations_in_test_set(start=START_DATE_TIME_IN_HK, end=END_DATE_TIME_IN_HK):
    result_objs = []
    exclude_dates = [datetime(2020, 1, 25), datetime(2020, 1, 27), datetime(2020, 1, 28)]
    start_datetime = datetime.strptime(start, '%Y-%m-%d-%H-%M-%S')
    looping_datetime = start_datetime
    end_datetime = datetime.strptime(end, '%Y-%m-%d-%H-%M-%S')

    while looping_datetime.date() < end_datetime.date():
        # Exclude Holidays
        for exclude_date in exclude_dates:
            if exclude_date.date() == looping_datetime.date():
                looping_datetime = looping_datetime + timedelta(days=1)
                continue

        # Exclude Week Days
        if looping_datetime.weekday() == 5 or looping_datetime.weekday() == 6:
            looping_datetime = looping_datetime + timedelta(days=1)
            continue

        while looping_datetime.time() < end_datetime.time():
            print("recommending: " + str(looping_datetime))
            result_obj = get_recommendations(date=looping_datetime.strftime('%Y-%m-%d-%H-%M-%S'))
            predict_col.insert(result_obj)
            result_objs.append(result_obj)
            looping_datetime = looping_datetime + timedelta(hours=1)

        looping_datetime = looping_datetime + timedelta(days=1)
        looping_datetime = looping_datetime.replace(hour=start_datetime.hour, minute=start_datetime.minute,
                                                    second=start_datetime.second)
    return result_objs


result_objs = get_recommendations_in_test_set()
#predict_col.insert_many(result_objs)
