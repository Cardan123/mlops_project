import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import os
import argparse
import json
import joblib
from dotenv import load_dotenv

# Set the default MLflow tracking URI, with environment variable as an option
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ModelEvaluator:
    """
    Class to evaluate a trained model by loading test data and computing metrics,
    then logging metrics and predictions to MLflow.
    """
    def __init__(self, X_test_path, y_test_path, model_path, experiment_name="Testing"):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path
        self.experiment_name = experiment_name

    def load_data(self):
        """Load test data for evaluation from specified CSV paths."""
        try:
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path)
            return X_test, y_test
        except FileNotFoundError as e:
            print(f"Error loading test data: {e}")
            raise

    def load_model(self):
        """Load a trained model from a file using joblib."""
        try:
            model = joblib.load(self.model_path)
            return model
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            raise
        except joblib.JoblibException as e:
            print(f"Error in joblib: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """
        Make predictions on test data and calculate evaluation metrics.

        Parameters:
        model: Trained model to evaluate
        X_test: Features of the test dataset
        y_test: Target values of the test dataset

        Returns:
        mse: Mean Squared Error
        r2: R-squared
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        y_pred: Model predictions
        """
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return mse, r2, mae, rmse, y_pred

    def log_metrics(self, mse, r2, mae, rmse):
        """Log evaluation metrics to MLflow."""
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)

    def log_predictions(self, y_test, y_pred):
        """Log predictions by saving to a CSV file and logging as an MLflow artifact."""
        pred_df = pd.DataFrame({"actual": y_test.values.ravel(), "predicted": y_pred})
        os.makedirs("data/predictions", exist_ok=True)
        pred_df.to_csv("data/predictions/predictions.csv", index=False)
        mlflow.log_artifact("data/predictions/predictions.csv")

    def save_metrics(self, metrics, output_path):
        """Save evaluation metrics as a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f)

    def ensure_experiment_active(self):
        """
        Ensure the MLflow experiment is active, creating or restoring it as necessary.

        Returns:
        experiment_id: ID of the MLflow experiment
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment:
            if experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                print(f"Experiment '{self.experiment_name}' restored.")
            return experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            print(f"Experiment '{self.experiment_name}' created.")
            return experiment_id

    def run_evaluation(self):
        """Execute the full evaluation pipeline, logging results to MLflow."""
        experiment_id = self.ensure_experiment_active()
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run():
            X_test, y_test = self.load_data()
            model = self.load_model()
            mse, r2, mae, rmse, y_pred = self.evaluate_model(model, X_test, y_test)
            self.log_metrics(mse, r2, mae, rmse)
            self.log_predictions(y_test, y_pred)

            # Save metrics to a JSON file for reference
            metrics = {
                "mse": mse,
                "r2": r2,
                "mae": mae,
                "rmse": rmse
            }
            os.makedirs("metrics", exist_ok=True)
            self.save_metrics(metrics, "metrics/evaluation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--X_test_path", type=str, help="Path to X_test.csv")
    parser.add_argument("--y_test_path", type=str, help="Path to y_test.csv")
    parser.add_argument("--model_path", type=str, help="Path to the best model file")
    args = parser.parse_args()

    # Initialize ModelEvaluator with the provided arguments and run the evaluation
    me = ModelEvaluator(args.X_test_path, args.y_test_path, args.model_path)
    me.run_evaluation()
