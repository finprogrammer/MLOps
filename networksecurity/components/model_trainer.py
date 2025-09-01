import os
import sys
from urllib.parse import urlparse
import tempfile, joblib
#os.environ["MLFLOW_ENABLE_LOGGED_MODELS"] = "false"

import mlflow
from sklearn.tree import DecisionTreeClassifier

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


#os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
##os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
#os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
_mlflow_user = os.getenv("MLFLOW_TRACKING_USERNAME")
_mlflow_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")


if _mlflow_uri is not None:
    os.environ["MLFLOW_TRACKING_URI"] = _mlflow_uri
if _mlflow_user is not None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = _mlflow_user
if _mlflow_pass is not None:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _mlflow_pass

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        mlflow.set_registry_uri(mlflow.get_tracking_uri())
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)

            scheme = urlparse(mlflow.get_tracking_uri()).scheme
            if scheme == "file":
                # Local store: full sklearn flavor is fine
                mlflow.sklearn.log_model(best_model, artifact_path="model")
            else:
                # Remote store (DagsHub): log plain pickle to avoid unsupported endpoint
                with tempfile.TemporaryDirectory() as d:
                    p = os.path.join(d, "model.pkl")
                    joblib.dump(best_model, p)
                    mlflow.log_artifact(p, artifact_path="model")

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "DecisionTreeClassifier": DecisionTreeClassifier()
        }

        params = {
            "DecisionTreeClassifier": {}
        }

        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        # Fit the chosen model before using it
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
