import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load environment variables and set up MLflow tracking URI
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelTrainer:
    """
    A class for training multiple models, logging performance metrics to MLflow, and registering the best model.
    """
    def __init__(self, X_train_path: str, y_train_path: str, params_path: str, experiment_name: str = "Training"):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.params_path = params_path
        self.experiment_name = experiment_name

    def load_data(self):
        """Load training data from CSV files."""
        X_train = pd.read_csv(self.X_train_path)
        y_train = pd.read_csv(self.y_train_path)
        return X_train, y_train

    def load_params(self):
        """Load model parameters from a YAML file."""
        with open(self.params_path, 'r') as fd:
            params = yaml.safe_load(fd)
        return params['train_model']['models']

    def train_and_evaluate(self, X_train, y_train, models_params):
        """Train and evaluate each model, logging results to MLflow, and track the best model."""
        best_model = None
        best_score = float('inf')
        best_model_name = ""
        best_params = {}
        best_run_id = None

        mlflow.set_experiment(self.experiment_name)

        for model_name, param_grid in models_params.items():
            model = self.get_model_instance(model_name)
            if model is None:
                continue

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train.values.ravel())
            mean_mse = -grid_search.best_score_
            print(f"{model_name} Best MSE: {mean_mse}")

            run_id = self.log_model_and_metrics(
                grid_search.best_estimator_, X_train, y_train, model_name, grid_search.best_params_, mean_mse
            )

            if mean_mse < best_score:
                best_score = mean_mse
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                best_params = grid_search.best_params_
                best_run_id = run_id 

        return best_model, best_model_name, best_params, best_run_id

    def get_model_instance(self, model_name):
        """Return the model instance based on model name."""
        if model_name == 'LinearRegression':
            return LinearRegression()
        elif model_name == 'Ridge':
            return Ridge()
        elif model_name == 'Lasso':
            return Lasso()
        elif model_name == 'DecisionTreeRegressor':
            return DecisionTreeRegressor(random_state=42)
        elif model_name == 'KNeighborsRegressor':
            return KNeighborsRegressor()
        else:
            print(f"Model {model_name} is not recognized.")
            return None

    def log_model_and_metrics(self, model, X_train, y_train, model_name, best_params, mean_mse):
        """Log model, parameters, and performance metrics to MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=model_name):
            run_id = mlflow.active_run().info.run_id
            mlflow.log_params(best_params)
            mlflow.set_tag("model_name", model_name)

            y_pred_train = model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            residuals = y_train.values.ravel() - y_pred_train
            residuals_std_dev = residuals.std()

            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("cv_mean_mse", mean_mse)
            mlflow.log_dict({'residuals': residuals.tolist()}, "residuals.json")
            mlflow.log_metric("residuals_std_dev", residuals_std_dev)

            input_example = X_train.head(1)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)

            return run_id

    def tag_best_model(self, best_run_id):
        """Tag the best model in MLflow as 'best_model'."""
        client = mlflow.tracking.MlflowClient()
        client.set_tag(best_run_id, "best_model", "True")
        print(f"Best model run ID: {best_run_id} tagged as best_model.")

    def register_best_model(self, best_run_id, model_name="BestModel"):
        """Register the best model in the MLflow Model Registry and transition it to production."""
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        print(f"Registered model '{model_name}' with version {result.version}")

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        print(f"Model '{model_name}' version {result.version} transitioned to 'Production' stage.")

    def save_best_model(self, model):
        """Save the best model to a file."""
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/best_model.pkl")

    def run(self):
        """Run the entire model training, selection, and logging workflow."""
        X_train, y_train = self.load_data()
        models_params = self.load_params()
        best_model, best_model_name, best_params, best_run_id = self.train_and_evaluate(X_train, y_train, models_params)
        print(f"Best model: {best_model_name} with params: {best_params}")
        self.save_best_model(best_model)
        self.tag_best_model(best_run_id)
        self.register_best_model(best_run_id, model_name="BestModel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--X_train_path", type=str, help="Path to X_train.csv")
    parser.add_argument("--y_train_path", type=str, help="Path to y_train.csv")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    mt = ModelTrainer(args.X_train_path, args.y_train_path, args.params, experiment_name="Training")
    mt.run()
