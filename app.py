from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import nltk
import string
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
import shutil
import tempfile
from fastapi.middleware.cors import CORSMiddleware

from extract_data import extract_paper_details

# Init

app = FastAPI()

nltk.download('stopwords')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))


# Request schema
class InputText(BaseModel):
    text: str


# Preprocess  (Input text undergoes the same preprocessing as the training text data)
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# Prediction
#def predict(text):
#    clean = preprocess(text)
#    vec = vectorizer.transform([clean])
#    pred = model.predict(vec)[0]
#    probs = model.predict_proba(vec)[0]
#    conf = float(max(probs))
#    return pred, conf
def predict(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    conf = float(max(probs))

    # confidence for each category
    all_scores = {
        category: float(prob)
        for category, prob in zip(model.classes_, probs)
    }
    return pred, conf, all_scores

# LIME (Explainable AI in the local scope)
explainer = LimeTextExplainer(class_names=model.classes_)

def predict_proba(texts):
    texts_clean = [preprocess(t) for t in texts]
    vectors = vectorizer.transform(texts_clean)
    return model.predict_proba(vectors)


# API Route

@app.post("/extract")
def extract_meta(file: UploadFile = File(...)):
    # validate
    if not file.filename.endswith(".pdf"):
        return { "error": "Only PDF Files are allowed"}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    
    try:
        result = extract_paper_details(temp_path)

        return result
    except Exception as e:
        return { "error": str(e)}

@app.post("/predict")
def predict_api(input: InputText):
    text = input.text

    pred, conf, all_scores = predict(text)

    label_index = list(model.classes_).index(pred)

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=8,
        labels=[label_index]
    )

    explanation = exp.as_list(label=label_index)

    return {
        "category": pred,
        "confidence": conf,
        "scores": all_scores,
        "explanation": [
            {"word": w, "weight": float(weight)}
            for w, weight in explanation
        ]
    }

