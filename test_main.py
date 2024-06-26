from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_joy():
    response = client.post(
        "/predict/",
        json={
            "text": "Будьте благополучны и радостны, и передайте мне на расстоянии, чтобы я знал, что с Вами все хорошо."
        },
    )
    json_data = response.json()
    assert response.status_code == 200
    assert "joy" in str(json_data[0])


def test_predict_no_emotion():
    response = client.post(
        "/predict/",
        json={"text": "Данный фрагмент текста не содержит абсолютно никаких эмоций"},
    )
    json_data = response.json()
    assert response.status_code == 200
    assert "no_emotion" in str(json_data[0])


def test_predict_sadness():
    response = client.post("/predict/", json={"text": "Грусть-тоска меня съедает"})
    json_data = response.json()
    assert response.status_code == 200
    assert "sadness" in str(json_data[0])


def test_predict_surprise():
    response = client.post(
        "/predict/", json={"text": "Нифига себе, неужели так тоже бывает!"}
    )
    json_data = response.json()
    assert response.status_code == 200
    assert "surprise" in str(json_data[0])


def test_predict_fear():
    response = client.post(
        "/predict/", json={"text": "Как-то стрёмно, давай свалим отсюда?"}
    )
    json_data = response.json()
    assert response.status_code == 200
    assert "fear" in str(json_data[0])


def test_predict_anger():
    response = client.post("/predict/", json={"text": "Бесишь меня, гад"})
    json_data = response.json()
    assert response.status_code == 200
    assert "anger" in str(json_data[0])
