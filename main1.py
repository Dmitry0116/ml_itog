from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str

app = FastAPI()
clf = pipeline(
    task = 'text-classification', 
    model = 'cointegrated/rubert-tiny2-cedr-emotion-detection')

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    """Распознование эмоций в тексте на русском языке"""
    return clf(item.text)
