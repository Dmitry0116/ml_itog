from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_joy():
    response = client.post("/predict/",
                           json={"text": "Будьте благополучны и радостны, и передайте мне на расстоянии, чтобы я знал, что с Вами все хорошо."})
    json_data = response.json()
    print (json_data[0].label)
    assert response.status_code == 200
    assert json_data['label'] == 'joy'


def test_predict_no_emotion():
    response = client.post("/predict/",
                           json={"text": "Данный фрагмент текста не содержит абсолютно никаких эмоций"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'no_emotion'


def test_predict_sadness():
    response = client.post("/predict/",
                           json={"text": "Грусть-тоска меня съедает"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'sadness'


def test_predict_surprise():
    response = client.post("/predict/",
                           json={"text": "Нифига себе, неужели так тоже бывает!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'surprise'
    

def test_predict_fear():
    response = client.post("/predict/",
                           json={"text": "Как-то стрёмно, давай свалим отсюда?"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'fear'

def test_predict_anger():
    response = client.post("/predict/",
                           json={"text": "Бесишь меня, гад"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'anger' 
