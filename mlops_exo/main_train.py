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
from mlflow.models.signature import infer_signature
from config import DATA_RAW, DATA_PROCESSED, MODELS_DIR
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure MLflow for Codespaces - use absolute path to mlruns
project_root = Path(__file__).parent.parent
mlruns_path = project_root / "mlops_exo" / "mlruns"
mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")


def main():
    """
    Loads, prepares data and trains a model. All artefacts are saved in the models folder
    :return:
    """
    # load data and split train set
    print("----- Loading data")
    path_train_set = DATA_RAW / "train.csv"
    path_features_set = DATA_RAW / "features.csv"
    path_stores_set = DATA_RAW / "stores.csv"
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
    x_train.to_parquet(DATA_PROCESSED / "x_train_processed.parquet", index=True)
    x_val.to_parquet(DATA_PROCESSED / "x_val_processed.parquet", index=True)
    pd.DataFrame(y_train).to_parquet(DATA_PROCESSED / "y_train.parquet", index=True)
    pd.DataFrame(y_val).to_parquet(DATA_PROCESSED / "y_val.parquet", index=True)

    # features selection
    x_train = x_train.drop(columns=["Date", "IsHoliday", "Type"])
    x_val = x_val.drop(columns=["Date", "IsHoliday", "Type"])

    # train model
    print("\n----- Train model and make predictions")
    model, dict_params = train_model(x_train, y_train)
    pred_train = pd.Series(
        predict_with_model(x_train, model), name="prediction", index=x_train.index
    )
    pred_val = pd.Series(
        predict_with_model(x_val, model), name="prediction", index=x_val.index
    )

    # display metrics
    print("\n----- Evaluating model")
    print("-- Train set :")
    dict_metrics_train = compute_metrics(y_train, pred_train, set="train")
    print("-- Validation set :")
    dict_metrics_val = compute_metrics(y_val, pred_val, set="val")

    # save predictions and artefacts
    print("\n----- save model, predictions and artifacts")

    # save predictions
    pd.DataFrame(pred_train).to_parquet(DATA_PROCESSED / "pred_train.parquet", index=True)
    pd.DataFrame(pred_val).to_parquet(DATA_PROCESSED / "pred_val.parquet", index=True)

    # save local artefacts
    joblib.dump(cleaner, MODELS_DIR / "cleaner.pkl")
    joblib.dump(features_transformer, MODELS_DIR / "features_transformer.pkl")
    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(DataCollector(), MODELS_DIR / "data_collector.pkl")

    # save model
    # Exercice 3.3 : lancer le run MLFlow et assignez un nom à l'exérimentation
    # ------------------------------------------------------------------------------------
    mlflow.set_experiment("Exo MLOps : training random forest")

    with mlflow.start_run() as run:
        # ------------------------------------------------------------------------------------
        # Exercice 3.3 : enregistrer les paramètres et se trouvant dans dict_params
        # ------------------------------------------------------------------------------------
        mlflow.log_params(dict_params)

        # ------------------------------------------------------------------------------------
        # Exercice 3.3 : enregistrer les métriques dans dict_metrics_train et dict_metrics_val
        # ------------------------------------------------------------------------------------
        mlflow.log_metrics(dict_metrics_train)
        mlflow.log_metrics(dict_metrics_val)

        # ------------------------------------------------------------------------------------
        # Exercice 3.3 : enregistrer les artefacts
        # ------------------------------------------------------------------------------------
        mlflow.log_artifact(MODELS_DIR / "cleaner.pkl")
        mlflow.log_artifact(MODELS_DIR / "features_transformer.pkl")
        mlflow.log_artifact(DATA_PROCESSED / "x_train_processed.parquet")
        mlflow.log_artifact(DATA_PROCESSED / "x_val_processed.parquet")
        mlflow.log_artifact(DATA_PROCESSED / "y_train.parquet")
        mlflow.log_artifact(DATA_PROCESSED / "y_val.parquet")

        # ------------------------------------------------------------------------------------
        # Exercice 4.1.A : enregistrer le modèle et la signature
        # ------------------------------------------------------------------------------------
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Run ID: {run.info.run_id}")
        signature = infer_signature(x_train, pred_train)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=x_train.iloc[0:1])
        # ------------------------------------------------------------------------------------
        
        # Exercice 4.1.C : Enregistrer les artefact manquants
        # ------------------------------------------------------------------------------------
        mlflow.log_artifact(path_features_set)
        mlflow.log_artifact(path_stores_set)
        mlflow.log_artifact(MODELS_DIR / "data_collector.pkl")
        # ------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()
