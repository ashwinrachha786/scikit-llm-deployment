from fastapi import FastAPI, Body, Request, UploadFile, File
import pandas as pd
import numpy as np
from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier
import os
from fastapi.middleware.cors import CORSMiddleware

from environs import Env

from pydantic import BaseModel

class TextModel(BaseModel):
    text: str


dotenv_path = "/workspaces/TransactWise/transact-wise/.env"
# dotenv.load_dotenv(dotenv_path)
env = Env()
env.read_env()

OPENAI_API_KEY = env.str("OPENAI_API_KEY")

df = pd.read_csv('Sample_Full_Transactions_Positive.csv')
X, y = df["Description"], df["Category"]

# Initialize Scikit-LLM model (move inside app for better access)
SKLLMConfig.set_openai_key(os.environ["OPENAI_API_KEY"])
clf = ZeroShotGPTClassifier(openai_model="gpt-4")
clf.fit(X, y)


app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/api/healthchecker")
def healthchecker():
    return {"status": "success", "message": "Integrate FastAPI Framework with Next.js"}

# Prediction endpoint for text classification
@app.post("/api/classify")
async def classify_text(text_model: TextModel):
    text = text_model.text
    if not text:
        return {"error": "Text cannot be empty."}

    predicted_label = clf.predict([text])[0]
    return {"predicted_label": predicted_label}


@app.post("/api/classify-file")
async def classify_text(file: UploadFile = File(...)):
    dataframe = pd.read_csv(file.file)
    # Assume dataframe has a 'Description' column for classification
    descriptions = dataframe['Description'].tolist()
    
    predicted_categories = clf.predict(descriptions)

    # Add predictions to dataframe and return as JSON
    dataframe['PredictedCategory'] = predicted_categories
    return dataframe.to_dict(orient='records')

