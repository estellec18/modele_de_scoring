from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import shap
import mpld3

# uvicorn main:app --reload in terminal
# http://127.0.0.1:8000/ => endpoint/route/api

class UserInput(BaseModel):
    num_client: int
    feat: str

app = FastAPI()

def load():
    """fonction qui charge le modele et l'explainer et prépare le dataset sur lequel va porter l'api"""
    model = joblib.load("best_xgb.joblib")
    explainer = joblib.load("explainer_xgb.joblib")
    scaler = model["scaler"]
    test = pd.read_csv('./data/clean/test.csv', index_col=0)
    test.set_index("SK_ID_CURR", inplace=True)
    train = pd.read_csv('./data/clean/training.csv', index_col=0)
    train.drop("TARGET", axis=1, inplace=True)
    train.set_index("SK_ID_CURR", inplace=True)
    concat = pd.concat([train, test])
    sub_df = concat.sample(frac=0.07, replace=False, random_state=42)
    features = pd.read_csv('./data/clean/features.csv', index_col=0)
    return model, explainer, scaler, sub_df, features

def create_df_proba(df, seuil:float):
    proba = model.predict_proba(df)
    df_proba = pd.DataFrame({'client_num':df.index, "proba_no_default":proba.transpose()[0], "proba_default":proba.transpose()[1]})
    df_proba["prediction"] = np.where(df_proba["proba_default"] > seuil, 1, 0)
    return df_proba

model, explainer, scaler, data, features = load()
seuil_predict = 0.56
pred_data = create_df_proba(data, seuil_predict)

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.post("/id_client")
def get_list_id():
    list_id = data.index.to_list()
    list_feature = features["Row"].to_list()
    return {"list_id":list_id,
            "list_feat":list_feature}

@app.post("/predict")
def predict(item:UserInput):
    results = pred_data[pred_data["client_num"]==item.num_client]
    if results["prediction"].values==0:
        verdict="Demande de crédit acceptée ✅"
    else:
        verdict="Demande de crédit refusée ⛔"
    proba = f"Nous estimons la probabilité de default du client à : {results['proba_default'].values[0]*100:.2f}%"
    return {"verdict":verdict, 
            "proba":proba}

@app.post("/gauge")
def gauge(item:UserInput):
    value = pred_data[pred_data["client_num"]==item.num_client]["proba_default"].values[0]
    if value > seuil_predict:
        color = "orange"
    else:
        color = "green"
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number+delta",
        title = {'text': "Probabilité de défault"},
        delta = {'reference': seuil_predict, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {'axis': {'range': [None, 1]},
                'bar' : {'color':color},
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': seuil_predict}}))
    fig_html = fig.to_html(fig) # pas trouver d'autres moyen que de convertir en html
    return{"fig":fig_html}


@app.post("/description")
def get_description(item:UserInput):
    result = features[features["Row"]==item.feat]["Description"].values[0]
    return {"description":str(result)}


def st_shap(plot, height=None):
   shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
   return shap_html

@app.post("/explanation")
def get_explanation(item:UserInput):
    scaled_data = scaler.transform(data)
    idx = pred_data.index[pred_data["client_num"]==item.num_client].values[0]
    data_idx = scaled_data[idx].reshape(1,-1)
    shap_values = explainer.shap_values(data_idx, l1_reg="aic")
    fig = st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], data_idx[0],feature_names=data.columns))
    
    df_shap = pd.DataFrame(shap_values[1], columns=data.columns)
    list_feat = []
    for i in range(9):
        max_col = df_shap.abs().max().idxmax()
        list_feat.append(max_col)
        df_shap.drop(max_col, axis=1, inplace=True)
    df_feat = data[data.index==item.num_client][list_feat].transpose().round(2)

    list_feat = df_feat.index.to_list()
    descr = []
    for feat in list_feat:
        desc = features[features["Row"]==feat]["Description"].values[0]
        descr.append(desc)
    
    df_feat.insert(column="Description", value=descr, loc=1)

    return{"df_feat":df_feat,
           "fig":fig}


@app.post("/perso_info")
def get_perso(item:UserInput):
    df = data.reset_index()
    
    gender = int(df[df["SK_ID_CURR"]==item.num_client]["CODE_GENDER"].values[0])
    nb_child = int(df[df["SK_ID_CURR"]==item.num_client]["CNT_CHILDREN"].values[0])
    income_amount = float(df[df["SK_ID_CURR"]==item.num_client]["AMT_INCOME_TOTAL"].values[0])
    credit = float(df[df["SK_ID_CURR"]==item.num_client]["AMT_CREDIT"].values[0])
    
    list_col = [col for col in df if col.startswith("NAME_INCOME_TYPE") or col.startswith("NAME_FAMILY_STATUS")]
    input = df[df["SK_ID_CURR"]==item.num_client][list_col]
    list_comp = [c for c in input.columns if input[c].values[0] == 1]
    income_type = str(list_comp[0].rsplit('_')[-1])
    family = str(list_comp[1].rsplit('_')[-1])

    return{"gender": gender,
           "nb_child": nb_child,
           "income_amount": income_amount,
           "credit": credit,
           "income_type":income_type,
           "family":family}


# takes too long ou message d'erreur du mac
# @app.post("/for_waterfall")
# def get_waterfall(item:UserInput):
#     scaled_data = scaler.transform(data)
#     idx = pred_data.index[pred_data["client_num"]==item.num_client].values[0]
#     data_idx = scaled_data[idx].reshape(1,-1)
#     shap_values = explainer.shap_values(data, l1_reg="aic")
#     exp = shap.Explanation(shap_values[1], explainer.expected_value[1], data_idx, feature_names=data.columns)
#     fig = st_shap(shap.plots.waterfall(exp[0]))
#     return{"waterfall":fig}