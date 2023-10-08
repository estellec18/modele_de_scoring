import pytest
import pandas as pd

from app import load_data
from app import def_seuil
from app import get_prediction


# s'assurer que le dataframe retourné n'est pas vide
def test_not_empty():
    assert load_data().shape != (0,0)

# s'assurer qu'il n'y a pas de ligne en doublon dans le dataset retourné
@pytest.fixture
def duplicated_load_data() -> pd.DataFrame:
    df_toload = load_data()
    df_toload_dup = df_toload[df_toload.duplicated()]
    return df_toload_dup

def test_duplicate_in_data(duplicated_load_data:callable):
    assert duplicated_load_data.empty

# s'assurer que les nums clients "SK_ID_CURR" constituent bien l'index du dataset retourné
@pytest.fixture
def load_data_idx() -> str:
    df_toload = load_data()
    index = df_toload.index.name
    return index

def test_numclient_in_data(load_data_idx:callable):
    assert "SK_ID_CURR" == load_data_idx


# test que tel num client que je sais etre 1 est bien 1
def prediction_positive():
    list_id = [312338,339038,349304,200971]
    pred = []
    for id in list_id:
        verdict, proba = get_prediction(id)
        pred.append(verdict)
    return set(pred)

def test_application_accepted():
    assert len(prediction_positive())==1
    assert prediction_positive()=={"Demande de crédit acceptée ✅"}


# test que tel num client que je sais etre 0 est bien 0
def prediction_negative():
    list_id = [379119,313656,141215,324620]
    pred = []
    for id in list_id:
        verdict, proba = get_prediction(id)
        pred.append(verdict)
    return set(pred)

def test_application_rejected():
    assert len(prediction_negative())==1
    assert prediction_negative()=={"Demande de crédit refusée ⛔"}

