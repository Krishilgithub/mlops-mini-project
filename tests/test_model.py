import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "campusx-official"
        repo_name = "mlops-project-2"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        cls.new_model_name = "my_model"

        try:
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
            if cls.new_model_version is None:
                raise RuntimeError("No model version found in registry for 'my_model'")

            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except MlflowException as e:
            raise RuntimeError(f"Failed to load model from MLflow: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading model: {e}")

        # Load the vectorizer
        try:
            cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("Vectorizer file not found at 'models/vectorizer.pkl'")
        except Exception as e:
            raise RuntimeError(f"Failed to load vectorizer: {e}")

        # Load holdout test data
        try:
            cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Test data not found at 'data/processed/test_bow.csv'")
        except Exception as e:
            raise RuntimeError(f"Failed to load test data: {e}")

    @staticmethod
    def get_latest_model_version(model_name):
        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None
            latest = max(versions, key=lambda v: int(v.version))
            return latest.version
        except Exception as e:
            raise RuntimeError(f"Error fetching model version: {e}")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())

        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')


if __name__ == "__main__":
    unittest.main()
