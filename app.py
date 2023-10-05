import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
  path = '/Users/estellecampos/Documents/Reconversion/Formation OpenClassroom/Projet_7/data/clean/'
  filename = "test.csv"
  data = pd.read_csv(path + filename, index_col=0)
  data.set_index("SK_ID_CURR", inplace=True)
  return data

data = load_data()

model = joblib.load("best_logreg.joblib")
explainer = joblib.load("explainer.joblib")
scaler = model["scaler"]
prediction = model.predict(data)
proba = model.predict_proba(data)
pred_data = pd.DataFrame({'client_num':data.index, 'prediction':prediction, "proba_no_default":proba.transpose()[0], "proba_default":proba.transpose()[1]})


def get_prediction(num_client):
    results = pred_data[pred_data["client_num"]==num_client]
    if results["prediction"].values==0:
        verdict="Demande de crédit acceptée 🌟"
    else:
        verdict="Demande de crédit refusée ⛔"
    proba = f"Nous estimons la probabilité de default du client à : {results['proba_default'].values[0]*100:.2f}%"
    return verdict, proba


def get_explanation(num_client):
    scaled_data = scaler.transform(data)
    idx = pred_data.index[pred_data["client_num"]==num_client].values[0]
    data_idx = scaled_data[idx].reshape(1,-1)
    shap_values = explainer.shap_values(data_idx, l1_reg="aic")
    fig = shap.force_plot(explainer.expected_value[1], shap_values[1][0], data_idx[0], feature_names=data.columns, matplotlib=True)
    df_shap = pd.DataFrame(shap_values[1], columns=data.columns)
    list_feat = []
    for i in range(10):
        max_col = df_shap.max().idxmax()
        list_feat.append(max_col)
        df_shap.drop(max_col, axis=1, inplace=True)
    df_feat = data[data.index==num_client][list_feat].transpose().round(2)
    return(fig, df_feat)


# Configures the default settings of the page (must be the first streamlit command and must be set once)
st.set_page_config(page_title="Prédiction de la capacité de remboursement d'un demandeur de prêt",
                   page_icon="🏦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# title and description
with st.container():
  st.title("Prédiction de la capacité de remboursement d'un demandeur de prêt")
  st.markdown("*Outil de prédiction développé dans le cadre du projet 7 du parcours OC Data Science*")
  st.markdown('##')
  with st.container():
    option = st.selectbox(
      "Veuillez spécifier le numéro d'identification du demandeur de prêt",
      (data.index))
    if st.button("Prediction"):
       verdict, proba = get_prediction(option)
       st.write(verdict)
       st.write(proba)
  st.markdown('##')
  with st.container():
    st.write("Une fois la prédiction effectuée, obtenez le détail en cliquant ci-dessous")
    if st.button("Détail"):
       fig, df_feat = get_explanation(option)
       st.pyplot(fig)
       st.dataframe(df_feat)
  st.markdown('#')
  with st.container():
     st.write("❗Cet outil permet d'assister à la prise de décision et doit être utilisé conjointement avec une analyse approfondie réalisée par un professionel.❗")

# from terminal : streamlit run app.py
