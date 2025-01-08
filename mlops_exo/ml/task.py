import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a RandomForestRegressor on the train dataset
    :param x_train: (DataFrame) containing the train features
    :param y_train: (Series) containing the train target
    :return: (RandomForestRegressor) trained model
    """
    # features selection
    print("features selection")

    print("training random forest")
    n_estimators = 30
    max_depth = 20
    min_samples_split = 10
    random_state = 42
    dict_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "random_state": random_state,
    }

    # print example
    print("observation example")
    print(x_train.iloc[0].to_json())

    # entrainement du modèle
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model, dict_params


def predict_with_model(x_test: pd.DataFrame, model: RandomForestRegressor):
    """
    Generates predictions on the test dataset
    :param x_test: (DataFrame) containing the test features
    :param model: (RandomForestRegressor) trained model
    :return: (Series) containing the predictions
    """
    # x_test = x_test.drop(columns=["Date", "IsHoliday", "Type"])
    return model.predict(x_test)
