import streamlit as st
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

st.title("Приложение определяет эмоции в предложениях на русском языке.")

form = st.form(key="sentiment-form")
user_input = form.text_area("Введите Ваш текст")
submit = form.form_submit_button("Определить")

if submit:
    clf = pipeline(
        task="text-classification",
        model="cointegrated/rubert-tiny2-cedr-emotion-detection",
    )

    result = clf(user_input)

    for res in result:
        emotion = res["label"]
        result = res["score"]
        st.success(f"{emotion} sentiment (score: {result})")


class Item(BaseModel):
    text: str


app = FastAPI()
clf = pipeline(
    task="text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection"
)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    """Распознование эмоций в тексте на русском языке"""
    return clf(item.text)
