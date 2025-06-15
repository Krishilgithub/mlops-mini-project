import mlflow
import os

# Replace this with your actual DAGSHUB_PAT
dagshub_pat = "6672019347d852481d31d23c3261fa659b8ad4af"

# Set credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_pat
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_pat

# Set DAGsHub tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Krishilgithub/mlops-mini-project.mlflow")

# Init client
client = mlflow.MlflowClient()

# Check if model exists
model_name = "my_model"

try:
    print(f"🔍 Fetching details for model: {model_name}")
    model_info = client.get_registered_model(model_name)
    print(f"✅ Model found: {model_info.name}")
    
    versions = client.get_latest_versions(model_name)
    for v in versions:
        print(f"📦 Version: {v.version}, Stage: {v.current_stage}, Status: {v.status}")
except Exception as e:
    print("❌ Error fetching model:", str(e))
