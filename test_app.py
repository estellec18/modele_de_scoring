import pytest
import pandas as pd

from app import load_data, create_df_proba, get_prediction

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
    df_toload = load_data()
    pred_data = create_df_proba(df_toload, 0.56)
    list_id = [287819,222036,183296,368092]
    pred = []
    for id in list_id:
        verdict, proba = get_prediction(pred_data,id)
        pred.append(verdict)
    return set(pred)

def test_application_accepted():
    assert len(prediction_positive())==1
    assert prediction_positive()=={"Demande de crédit acceptée ✅"}


# test que tel num client que je sais etre 0 est bien 0
def prediction_negative():
    df_toload = load_data()
    pred_data = create_df_proba(df_toload, 0.56)
    list_id = [331687,337539,139992,338430]
    pred = []
    for id in list_id:
        verdict, proba = get_prediction(pred_data,id)
        pred.append(verdict)
    return set(pred)

def test_application_rejected():
    assert len(prediction_negative())==1
    assert prediction_negative()=={"Demande de crédit refusée ⛔"}

