import pandas as pd
import joblib
from gathering.task import DataCollector
from gathering.cleaning import DataCleaner
from features.task import FeaturesEngineering
from ml.task import train_model, predict_with_model
from ml.validation import split_train_and_val_sets, compute_metrics
import warnings
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
warnings.filterwarnings('ignore')


def main():
    """
    Loads, prepares data and trains a model. All artefacts are saved in the models folder
    :return:
    """
    # load data and split train set
    print("----- Loading data")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_train_set = os.path.join(current_dir, "../data/raw/train.csv")
    path_features_set = os.path.join(current_dir, "../data/raw/features.csv")
    path_stores_set = os.path.join(current_dir, "../data/raw/stores.csv")
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
    print("save x_train_processed and x_val_processed")
    print("save y_train and y_val")
    x_train.to_parquet(os.path.join(current_dir, "../data/processed/x_train_processed.parquet"), index=True)
    x_val.to_parquet(os.path.join(current_dir, "../data/processed/x_val_processed.parquet"), index=True)
    pd.DataFrame(y_train).to_parquet(os.path.join(current_dir, "../data/processed/y_train.parquet"), index=True)
    pd.DataFrame(y_val).to_parquet(os.path.join(current_dir, "../data/processed/y_val.parquet"), index=True)

    # features selection
    x_train = x_train.drop(columns=["Date", "IsHoliday", "Type"])
    x_val = x_val.drop(columns=["Date", "IsHoliday", "Type"])

    # train model
    print("\n----- Train model and make predictions")
    model, dict_params = train_model(x_train, y_train)
    pred_train = pd.Series(predict_with_model(x_train, model), name="prediction", index=x_train.index)
    pred_val = pd.Series(predict_with_model(x_val, model), name="prediction", index=x_val.index)

    # display metrics
    print("\n----- Evaluating model")
    print("-- Train set :")
    dict_metrics_train = compute_metrics(y_train, pred_train, set="train")
    print("-- Validation set :")
    dict_metrics_val = compute_metrics(y_val, pred_val, set="val")

    # save predictions and artefacts
    print("\n----- save model, predictions and artifacts")

    # save predictions
    pd.DataFrame(pred_train).to_parquet(os.path.join(current_dir, "../data/processed/pred_train.parquet"), index=True)
    pd.DataFrame(pred_val).to_parquet(os.path.join(current_dir, "../data/processed/pred_val.parquet"), index=True)

    # save local artefacts
    joblib.dump(cleaner, os.path.join(current_dir, "../models", "cleaner.pkl"))
    joblib.dump(features_transformer, os.path.join(current_dir, "../models", "features_transformer.pkl"))
    joblib.dump(model, os.path.join(current_dir, "../models", "model.pkl"))

    # save model
    # TODO - exercice 3.3 : lancer le run MLFlow et assignez un nom à l'exérimentation
    # ------------------------------------------------------------------------------------
    # Initialisation MLflow
    mlflow.set_tracking_uri(os.path.join(current_dir, "mlruns"))
    mlflow.set_experiment("Exo MLOps - Random Forest")
    with mlflow.start_run():
        artifact_uri = mlflow.get_artifact_uri()
    # ------------------------------------------------------------------------------------

        # TODO - exercice 3.3 : enregistrer les paramètres et se trouvant dans dict_params
        # ------------------------------------------------------------------------------------
        # Sauvegarde des paramètres
        # mlflow.log_params(dict_params)
        mlflow.log_param("max_depth", dict_params["max_depth"])
        mlflow.log_param("n_estimators", dict_params["n_estimators"])
        mlflow.log_param("min_samples_split", dict_params["min_samples_split"])
        mlflow.log_param("random_state", dict_params["random_state"])
        # ------------------------------------------------------------------------------------

        # TODO - exercice 3.3 : enregistrer les métriques dans dict_metrics_train et dict_metrics_val
        # ------------------------------------------------------------------------------------
        # Sauvegarde des métriques
        # mlflow.log_metrics(dict_metrics_train)
        # mlflow.log_metrics(dict_metrics_val)
        mlflow.log_metric("mae_train", dict_metrics_train["mae_train"])
        mlflow.log_metric("mse_train", dict_metrics_train["mse_train"])
        mlflow.log_metric("mape_train", dict_metrics_train["mape_train"])
        mlflow.log_metric("mae_val", dict_metrics_val["mae_val"])
        mlflow.log_metric("mse_val", dict_metrics_val["mse_val"])
        mlflow.log_metric("mape_val", dict_metrics_val["mape_val"])
        # ------------------------------------------------------------------------------------

        # TODO - exercice 3.3 : enregistrer les artefacts
        # ------------------------------------------------------------------------------------
        # Sauvegarde des artefacts
        mlflow.log_artifact(os.path.join(current_dir, "../models", "cleaner.pkl"))
        mlflow.log_artifact(os.path.join(current_dir, "../models", "features_transformer.pkl"))
        mlflow.log_artifact(os.path.join(current_dir, "../data/raw/features.csv"))
        mlflow.log_artifact(os.path.join(current_dir, "../data/raw/stores.csv"))
        mlflow.log_artifact(os.path.join(current_dir, "../models", "model.pkl"))
        # ------------------------------------------------------------------------------------

        # TODO - exercice 4.1 : enregistrer le modèle et la signature
        # ------------------------------------------------------------------------------------
        # Sauvegarde du modèle
        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=x_train.iloc[0:1])
        # ------------------------------------------------------------------------------------



if __name__ == "__main__":
    main()
