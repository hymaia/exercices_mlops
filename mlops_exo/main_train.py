import mlflow.dspy
import pandas as pd
import joblib
from mlops_exo.gathering.task import DataCollector
from mlops_exo.gathering.cleaning import DataCleaner
from mlops_exo.features.task import FeaturesEngineering
from mlops_exo.ml.task import train_model
from mlops_exo.ml.validation import split_train_and_val_sets, compute_metrics
from mlops_exo.ml.utils import print_mlflow_artefact_uri
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Loads, prepares data and trains a model. All artefacts are saved in the models folder
    :return:
    """
    # load data and split train set
    print("----- Loading data")
    path_train_set = "../data/raw/train.csv"
    path_features_set = "../data/raw/features.csv"
    path_stores_set = "../data/raw/stores.csv"
    df_train = DataCollector().gather_data(
        path_train_set, path_features_set, path_stores_set
    )
    x_train, x_val, y_train, y_val = split_train_and_val_sets(df_train)

    # cleaning and features engineering
    print("\n----- Features engineering")
    cleaner = DataCleaner().fit(df_train)
    x_train = cleaner.transform(x_train)
    x_val = cleaner.transform(x_val)
    features_transformer = FeaturesEngineering().fit(x_train, y_train)
    x_train = features_transformer.transform(x_train)
    x_val = features_transformer.transform(x_val)
    print("save x_train_processed.xlsx")
    print("save x_val_processed.xlsx")
    x_train.head(100).to_excel("../data/processed/x_train_processed.xlsx", index=False)
    x_val.head(100).to_excel("../data/processed/x_val_processed.xlsx", index=False)

    # train model
    print("\n----- Train model and make predictions")
    model = train_model(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    # display metrics
    print("\n----- Evaluating model")
    print("-- Train set :")
    compute_metrics(y_train, pred_train, set="train")
    print("-- Validation set :")
    compute_metrics(y_val, pred_val, set="val")

    # save predictions and artefacts
    print("\n----- save model, predictions and artifacts")

    # save predictions
    print_mlflow_artefact_uri()
    pd.DataFrame(pred_train).head(100).to_excel("../data/processed/pred_train.xlsx", index=False)
    pd.DataFrame(pred_val).head(100).to_excel("../data/processed/pred_val.xlsx", index=False)

    # sve model TODO - exercice 4.1.A : enregistrer le modèle dans MLFlow
    joblib.dump(model, "../models/model.pkl")
    mlflow.sklearn.log_model(model, "model")

    # save artefacts # TODO - exercice 4.1.B : enregistrer les artifacts dans MLFlow
    joblib.dump(cleaner, "../models/cleaner.pkl")
    joblib.dump(features_transformer, "../models/features_transformer.pkl")
    mlflow.log_artifact("../models/cleaner.pkl")
    mlflow.log_artifact("../models/features_transformer.pkl")
    mlflow.log_artifact("../data/raw/features.csv")
    mlflow.log_artifact("../data/raw/stores.csv")


if __name__ == "__main__":
    main()