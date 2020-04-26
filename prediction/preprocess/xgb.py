from xgboost import XGBClassifier
from matplotlib import pyplot
from prediction.preprocess.preprocess import get_datasets
from prediction.cron.cron_stock import fetch_stock
from xgboost import plot_importance
from prediction.cron.cron_correlated_assets import fetch_currencies
from prediction.cron.cron_correlated_assets import fetch_composite_indices

# xgboost:
# ref: https://github.com/apachecn/ml-mastery-zh/blob/master/docs/xgboost/feature-importance-and-feature-selection-with-xgboost-in-python.md

# Having so many features we have to consider whether all of them are really indicative of the direction GS stock
# will take. For example, we included USD denominated LIBOR rates in the dataset because we think that changes in
# LIBOR might indicate changes in the economy, that, in turn, might indicate changes in the GS's stock behavior. But
# we need to test. There are many ways to test feature importance, but the one we will apply uses XGBoost,
# because it gives one of the best results in both classification and regression problems.
def get_feature_importance_data():
    train_input, train_labels, test_input, test_labels = get_datasets(fetch_stock(), fetch_composite_indices(),
                                                                      fetch_currencies())
    model = XGBClassifier()
    model.fit(train_input, train_labels)
    # plot feature importance
    plot_importance(model)
    pyplot.show()


get_feature_importance_data()