from flask import Flask, request, redirect, render_template
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle

mlflow.set_tracking_uri('https://dagshub.com/Krishilgithub/mlops-mini-project.mlflow')
dagshub.init(repo_owner='Krishilgithub', repo_name='mlops-mini-project', mlflow=True)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

#* Load model from model registry
model_name = "my_model"
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

app = Flask(__name__)

@app.route('/')
def name():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text")

    #* Clean
    text = normalize_text(text)

    #* bow
    features = vectorizer.transform([text])

    #* predict
    result = model.predict(features)

    return str(result[0])

app.run(debug=True)