import logging
import os
import re
import string
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import mlflow
import pickle
from mlflow.exceptions import MlflowException
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# Load environment variables from .env file
logger = logging.getLogger(__name__)  # Define logger early for use in dotenv loading
try:
    logger.info("📂 Loading .env file...")
    load_dotenv()  # Loads .env file from the current directory
    logger.info("✅ .env file loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load .env file: {e}")
    raise

# Set up logging after loading .env (so we can use logger for dotenv errors)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

# Download NLTK data
try:
    logger.info("📚 Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("✅ NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to download NLTK data: {e}")
    raise

# NLP Functions
def lemmatization(text):
    """Lemmatize the text."""
    try:
        logger.debug(f"Starting lemmatization for text: {text[:50]}...")
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        result = " ".join(text)
        logger.debug(f"Lemmatized text: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ Lemmatization failed: {e}")
        raise

def remove_stop_words(text):
    """Remove stop words from the text."""
    try:
        logger.debug(f"Removing stop words from text: {text[:50]}...")
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        result = " ".join(text)
        logger.debug(f"Text after removing stop words: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to remove stop words: {e}")
        raise

def removing_numbers(text):
    """Remove numbers from the text."""
    try:
        logger.debug(f"Removing numbers from text: {text[:50]}...")
        text = ''.join([char for char in text if not char.isdigit()])
        logger.debug(f"Text after removing numbers: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"❌ Failed to remove numbers: {e}")
        raise

def lower_case(text):
    """Convert text to lower case."""
    try:
        logger.debug(f"Converting text to lower case: {text[:50]}...")
        text = text.split()
        text = [word.lower() for word in text]
        result = " ".join(text)
        logger.debug(f"Text after lower case: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to convert to lower case: {e}")
        raise

def removing_punctuations(text):
    """Remove punctuations from the text."""
    try:
        logger.debug(f"Removing punctuations from text: {text[:50]}...")
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        logger.debug(f"Text after removing punctuations: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"❌ Failed to remove punctuations: {e}")
        raise

def removing_urls(text):
    """Remove URLs from the text."""
    try:
        logger.debug(f"Removing URLs from text: {text[:50]}...")
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        result = url_pattern.sub(r'', text)
        logger.debug(f"Text after removing URLs: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to remove URLs: {e}")
        raise

def normalize_text(text):
    """Normalize the text by applying all preprocessing steps."""
    try:
        logger.info(f"Starting text normalization for: {text[:50]}...")
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)
        logger.info(f"Normalized text: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"❌ Text normalization failed: {e}")
        raise

# Set up DagsHub credentials for MLflow tracking
logger.info("🔑 Setting up Dagshub credentials...")
dagshub_token = os.getenv("DAGSHUB_PAT")  # This will now be loaded from .env
if not dagshub_token:
    logger.error("❌ DAGSHUB_PAT environment variable is not set")
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set 😿")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
dagshub_url = "https://dagshub.com"
repo_owner = "Krishilgithub"
repo_name = "mlops-mini-project"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
logger.info(f"✅ MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# Flask app setup
app = Flask(__name__)

# Load model from MLflow registry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(MlflowException),
    reraise=True
)
def load_model(model_uri):
    """Load model with retry logic."""
    logger.info(f"📦 Loading model from URI: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

def get_latest_model_version(model_name):
    """Get the latest model version without using deprecated stages."""
    client = mlflow.MlflowClient()
    try:
        logger.info(f"🔍 Fetching versions for model: {model_name}")
        filter_string = f"name='{model_name}'"
        versions = client.search_model_versions(filter_string)
        if not versions:
            logger.warning(f"⚠️ No versions found for model {model_name}")
            return None
        latest_version = max(versions, key=lambda x: int(x.version))
        logger.info(f"✅ Latest version for {model_name}: {latest_version.version}")
        return latest_version.version
    except MlflowException as e:
        logger.error(f"❌ Failed to fetch model versions: {e}")
        raise

# Load the model
model_name = "my_model"
model = None
try:
    model_version = get_latest_model_version(model_name)
    if model_version is None:
        logger.error("❌ No valid model version found")
        raise ValueError("No valid model version found 😿")
    model_uri = f'models:/{model_name}/{model_version}'
    model = load_model(model_uri)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise

# Load the vectorizer
try:
    logger.info("📂 Loading vectorizer from models/vectorizer.pkl...")
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    logger.info("✅ Vectorizer loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load vectorizer: {e}")
    raise

# Routes
@app.route('/')
def home():
    logger.info("🏠 Rendering home page")
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("🔮 Starting prediction process")
        text = request.form['text']
        logger.debug(f"Received input text: {text[:50]}...")

        # Normalize text
        logger.info("🧹 Normalizing text...")
        normalized_text = normalize_text(text)
        logger.debug(f"Normalized text: {normalized_text[:50]}...")

        # Transform text using vectorizer
        logger.info("📊 Transforming text to BOW features...")
        features = vectorizer.transform([normalized_text])
        logger.debug(f"Features shape: {features.shape}")

        # Convert to DataFrame
        logger.info("📅 Converting features to DataFrame...")
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        logger.debug(f"Features DataFrame shape: {features_df.shape}")

        # Predict
        logger.info("🔍 Making prediction...")
        result = model.predict(features_df)
        logger.info(f"Prediction result: {result[0]}")

        # Render result
        logger.info("🎉 Rendering prediction result")
        return render_template('index.html', result=result[0])
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return render_template('index.html', result=None, error=f"Prediction failed: {str(e)} 😿")

# Bonus: Multiplication table route
@app.route('/multiply', methods=['GET', 'POST'])
def multiply():
    logger.info("🔢 Rendering multiplication table page")
    table = None
    number = None
    error = None
    if request.method == 'POST':
        try:
            logger.debug("Processing multiplication table request")
            number = int(request.form['number'])
            table = [f"{number} × {i} = {number * i}" for i in range(1, 11)]
            logger.info(f"Generated table for number {number}")
        except ValueError:
            logger.warning("Invalid input for multiplication table")
            error = "Please enter a valid integer! 😿"
    return render_template('multiply.html', table=table, number=number, error=error)

if __name__ == "__main__":
    logger.info("🚀 Starting Flask app...")
    app.run(debug=True, host="0.0.0.0", port=5000)