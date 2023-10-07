import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
  path = './data/clean/'
  filename = "test.csv"
  data = pd.read_csv(path + filename, index_col=0)
  data.set_index("SK_ID_CURR", inplace=True)
  return data

data = load_data()

model = joblib.load("best_xgb.joblib")
explainer = joblib.load("explainer_xgb.joblib")

scaler = model["scaler"]
proba = model.predict_proba(data)
pred_data = pd.DataFrame({'client_num':data.index, "proba_no_default":proba.transpose()[0], "proba_default":proba.transpose()[1]})

def def_seuil(df, seuil):
   df["prediction"] = np.where(df["proba_default"] > seuil, 1, 0)
   return df

seuil_predict = 0.56
pred_data = def_seuil(pred_data, seuil_predict)


def get_prediction(num_client):
    results = pred_data[pred_data["client_num"]==num_client]
    if results["prediction"].values==0:
        verdict="Demande de crédit acceptée ✅"
    else:
        verdict="Demande de crédit refusée ⛔"
    proba = f"Nous estimons la probabilité de default du client à : {results['proba_default'].values[0]*100:.2f}%"
    return verdict, proba

def gauge(num_client, seuil):
    value = pred_data[pred_data["client_num"]==num_client]["proba_default"].values[0]
    if value > seuil:
        color = "orange"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number+delta",
        title = {'text': "Probabilité de défault"},
        delta = {'reference': seuil, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {'axis': {'range': [None, 1]},
                'bar' : {'color':color},
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.56}}))
    
    return(fig)


def st_shap(plot, height=None):
   shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
   components.html(shap_html, height=height)

def get_explanation(num_client):
    scaled_data = scaler.transform(data)
    idx = pred_data.index[pred_data["client_num"]==num_client].values[0]
    data_idx = scaled_data[idx].reshape(1,-1)
    shap_values = explainer.shap_values(data_idx, l1_reg="aic")
    #st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], data_idx[0],feature_names=data.columns))
    #fig = shap.force_plot(explainer.expected_value[1], shap_values[1][0], data_idx[0], feature_names=data.columns, matplotlib=True)
    exp = shap.Explanation(shap_values[1], explainer.expected_value[1], data_idx, feature_names=data.columns)
    fig = shap.plots.waterfall(exp[0])
    df_shap = pd.DataFrame(shap_values[1], columns=data.columns)
    list_feat = []
    for i in range(9):
        max_col = df_shap.abs().max().idxmax()
        list_feat.append(max_col)
        df_shap.drop(max_col, axis=1, inplace=True)
    df_feat = data[data.index==num_client][list_feat].transpose().round(2)
    return(fig, df_feat)

def get_waterfall(num_client):
    scaled_data = scaler.transform(data)
    idx = pred_data.index[pred_data["client_num"]==num_client].values[0]
    data_idx = scaled_data[idx].reshape(1,-1)
    shap_values = explainer.shap_values(data, l1_reg="aic")
    exp = shap.Explanation(shap_values[1], explainer.expected_value[1], data_idx, feature_names=data.columns)
    fig = shap.plots.waterfall(exp[0])
    return fig

def position_global_distrib(det_feat, num_client):
    list_feat = list(det_feat.index)
    fig, axs = plt.subplots(3, 3, figsize=(15,11))
    i = 0
    j = 0
    for feat in list_feat:
        data[feat].plot(kind='hist', ax=axs[i,j])
        axs[i,j].set_title(feat)
        axs[i,j].axvline(x=det_feat.loc[feat,num_client], color='orange', linewidth=2)
        j+=1
        if j>2:
            i+=1
            j=0
    fig.tight_layout()
    return fig

def position_label_distrib(det_feat, num_client):
    pred = pred_data[pred_data["client_num"]==num_client]["prediction"].values[0]
    basis = pred_data[pred_data["prediction"]==pred]
    list_num = list(basis["client_num"])    
    new_data = data.loc[list_num]

    list_feat = list(det_feat.index)
    fig, axs = plt.subplots(3, 3, figsize=(15,11))
    i = 0
    j = 0
    for feat in list_feat:
        new_data[feat].plot(kind='hist', ax=axs[i,j])
        axs[i,j].set_title(feat)
        axs[i,j].axvline(x=det_feat.loc[feat,num_client], color='orange', linewidth=2)
        j+=1
        if j>2:
            i+=1
            j=0
    fig.tight_layout()
    return fig


## afficher seuil de décision ? sur une frise


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
       fig, ax = plt.subplots()
       fig = gauge(option, seuil_predict)
       st.plotly_chart(fig)
       fig, df_feat = get_explanation(option)
       col1, col2 = st.columns([2,1], gap="small")
       with col1:
        st.pyplot(fig)
       with col2:
        st.dataframe(df_feat)
       tab1, tab2 = st.tabs(["Positionnement du demandeur de crédit par rapport aux autres demandes", "Positionnement du demandeur de crédit par rapport aux autres demandes de sa catégorie"])
       with tab1:
        fig1 = position_global_distrib(df_feat, option)
        st.pyplot(fig1)
       with tab2:
        fig2 = position_label_distrib(df_feat, option)
        st.pyplot(fig2)
  st.markdown('#')
  with st.container():
     st.write("❗Cet outil permet d'assister à la prise de décision et doit être utilisé conjointement avec une analyse approfondie réalisée par un professionel.❗")



   

# from terminal : streamlit run app.py
