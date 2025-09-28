# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Try to import joblib and TensorFlow only if installed
try:
    import joblib
except ImportError:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

app = FastAPI(title="Emoji Translator")

# ---------------------------
# Pydantic model for input
# ---------------------------
class TextInput(BaseModel):
    text: str

# ---------------------------
# Load ML/DL models (if available)
# ---------------------------
ml_model_path = "models/baseline.pkl"
dl_model_path = "models/emoji_model.h5"

ml_model = None
dl_model = None

if joblib and os.path.exists(ml_model_path):
    ml_model = joblib.load(ml_model_path)
else:
    print(f"Warning: ML model not found at {ml_model_path}. Using dummy response.")

if load_model and os.path.exists(dl_model_path):
    dl_model = load_model(dl_model_path)
else:
    print(f"Warning: DL model not found at {dl_model_path}. Using dummy response.")

# ---------------------------
# /translate endpoint
# ---------------------------
@app.post("/translate")
def translate(input: TextInput):
    text = input.text

    # ML prediction
    if ml_model:
        # Replace this with actual ML prediction code
        ml_prediction = "ML_RESULT"
    else:
        ml_prediction = "N/A"

    # DL prediction
    if dl_model:
        # Replace this with actual DL prediction code
        dl_prediction = "DL_RESULT"
    else:
        dl_prediction = "🍕"  # dummy emoji

    return {
        "text_received": text,
        "emoji_prediction": dl_prediction,
        "ml_prediction": ml_prediction
    }
