from fastapi.testclient import TestClient
from main import app, load
import imblearn.pipeline
import sklearn.preprocessing
import shap
import pytest
import pandas as pd

client = TestClient(app=app)

def test_not_empty():
    model, explainer, scaler, data, features = load()
    assert type(model) == imblearn.pipeline.Pipeline
    assert type(explainer) == shap.explainers._kernel.Kernel
    assert type(scaler) == sklearn.preprocessing._data.MinMaxScaler
    assert data.shape != (0,0)
    assert features.shape != (0,0)

# s'assurer qu'il n'y a pas de ligne en doublon dans le dataset retourné
@pytest.fixture
def duplicated_load_data() -> pd.DataFrame:
    model, explainer, scaler, data, features = load()
    data_dup = data[data.duplicated()]
    return data_dup

def test_duplicate_in_data(duplicated_load_data:callable):
    assert duplicated_load_data.empty

# s'assurer que les nums clients "SK_ID_CURR" constituent bien l'index du dataset retourné
@pytest.fixture
def load_data_idx() -> str:
    model, explainer, scaler, data, features = load()
    index = data.index.name
    return index

def test_numclient_in_data(load_data_idx:callable):
    assert "SK_ID_CURR" == load_data_idx


# s'assurer que les données relatives aux ages des clients sont bien positives et supérieures à 18 (le pret à des personnes non majeures n'est pas légal)
def test_age():
    model, explainer, scaler, data, features = load()
    assert data["YEARS_BIRTH"].min() > 18


# s'assurer que les montants de demandes de crédit sont bien supérieurs à 0
def test_credit():
    model, explainer, scaler, data, features = load()
    assert data["AMT_CREDIT"].min() > 0

# les tests suivant sont relatifs au bon fonctionnement de la connection avec l'API

def test_can_call_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API"}

def test_predict_valid():
    payload = {"num_client": 144092, "feat": "str"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["verdict"] == "Demande de crédit acceptée ✅"

def test_predict_default():
    payload = {"num_client": 346201, "feat": "str"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["verdict"] == "Demande de crédit refusée ⛔"

def test_gauge():
    payload = {"num_client": 144092, "feat": "str"}
    response = client.post("/gauge", json=payload)
    assert response.status_code == 200

def test_update_gauge():
    payload = {"num_client": 144092, "feat": "str"}
    task_response = client.post("/gauge", json=payload)
    assert task_response.status_code==200
    
    new_payload = {"num_client": 346201, "feat": "str"}
    task_response_new = client.post("/gauge", json=new_payload)
    assert task_response_new.status_code==200

    assert task_response.json()["fig"] != task_response_new.json()["fig"]

def test_explanation():
    payload = {"num_client": 144092, "feat": "str"}
    response = client.post("/explanation", json=payload)
    assert response.status_code == 200

def test_perso_info():
    payload = {"num_client": 144092, "feat": "str"}
    response = client.post("/perso_info", json=payload)
    assert response.status_code == 200
    assert response.json() == { "gender": 1, "nb_child": 0, "income_amount": 256500.0, "credit": 609898.5, "income_type": "Working", "family": "Married"}



