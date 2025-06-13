import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.exceptions import MlflowException

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set 😿")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Krishilgithub"
        repo_name = "mlops-mini-project"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        if not cls.new_model_version:
            raise RuntimeError(f"No version found for model {cls.new_model_name} 🚫")

        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        try:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except MlflowException as e:
            raise RuntimeError(f"Failed to load model from {cls.new_model_uri}: {e} 😢")

        # Load the vectorizer
        try:
            cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("Vectorizer file 'models/vectorizer.pkl' not found 📁")

        # Load holdout test data
        try:
            cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Test data 'data/processed/test_bow.csv' not found 📊")

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        try:
            # Replace deprecated get_latest_versions with search_model_versions
            filter_string = f"name='{model_name}'"
            versions = client.search_model_versions(filter_string)
            if not versions:
                return None
            # Sort by version number (assuming version is numeric) and get the latest
            latest_version = max(versions, key=lambda x: int(x.version))
            return latest_version.version
        except MlflowException as e:
            print(f"Error fetching model versions for {model_name}: {e} ⚠️")
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, "Model should be loaded successfully 🎉")

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model
        prediction = self.new_model.predict(input_df)

        # Verify input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()),
                         "Input shape should match vectorizer features 📏")

        # Verify output shape
        self.assertEqual(len(prediction), input_df.shape[0], "Output should match input rows 📈")
        self.assertEqual(len(prediction.shape), 1, "Output should be 1D for binary classification 🎯")

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert performance
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy} 🌟')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision} ✨')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall} 🚀')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1} 🎈')

if __name__ == "__main__":
    unittest.main()