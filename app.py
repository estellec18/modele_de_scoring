import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

model = joblib.load("best_xgb.joblib")
explainer = joblib.load("explainer_xgb.joblib")
scaler = model["scaler"]

def load_data():
    test = pd.read_csv('./data/clean/test.csv', index_col=0)
    test.set_index("SK_ID_CURR", inplace=True)
    train = pd.read_csv('./data/clean/training.csv', index_col=0)
    train.drop("TARGET", axis=1, inplace=True)
    train.set_index("SK_ID_CURR", inplace=True)
    concat = pd.concat([train, test])
    sub_df = concat.sample(frac=0.07, replace=False, random_state=42)
    return sub_df

def create_df_proba(df, seuil):
    if seuil <= 1 and seuil >0:
        proba = model.predict_proba(df)
        df_proba = pd.DataFrame({'client_num':df.index, "proba_no_default":proba.transpose()[0], "proba_default":proba.transpose()[1]})
        df_proba["prediction"] = np.where(df_proba["proba_default"] > seuil, 1, 0)
        return df_proba
    else:
        print("veuillez choisir un seuil adéquat")
        return

data = load_data() # subsample composé de 10% des observations issues du training set + test set
seuil_predict = 0.56
pred_data = create_df_proba(data, seuil_predict) # df qui indique la proba de default / de non défault / la décision

def get_prediction(df, num_client):
    results = df[df["client_num"]==num_client]
    if results["prediction"].values==0:
        verdict="Demande de crédit acceptée ✅"
    else:
        verdict="Demande de crédit refusée ⛔"
    proba = f"Nous estimons la probabilité de default du client à : {results['proba_default'].values[0]*100:.2f}%"
    return verdict, proba

def gauge(df, num_client, seuil):
    value = df[df["client_num"]==num_client]["proba_default"].values[0]
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

def st_shap(plot, height=None): # plus utilisé
   shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
   components.html(shap_html, height=height)

def get_explanation(df, num_client):
    scaled_data = scaler.transform(data)
    idx = df.index[df["client_num"]==num_client].values[0]
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

def get_waterfall(df, num_client): #plus utilisé
    scaled_data = scaler.transform(data)
    idx = df.index[df["client_num"]==num_client].values[0]
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
    
    option = st.selectbox("Veuillez spécifier le numéro d'identification du demandeur de prêt",(data.index))
    if st.button("Prediction"):
        tab1, tab2, tab3 = st.tabs(["Prediction", "Positionnement du demandeur de crédit  \npar rapport aux autres demandes", "Positionnement du demandeur de crédit  \npar rapport aux autres demandes de sa catégorie"])
        
        with tab1:
            verdict, proba = get_prediction(pred_data, option)
            st.write(verdict)
            st.write(proba)
            fig, ax = plt.subplots()
            fig = gauge(pred_data, option, seuil_predict)
            st.plotly_chart(fig)
            fig, df_feat = get_explanation(pred_data, option)
            col1, col2 = st.columns([2,1], gap="small")
            with col1:
                st.pyplot(fig)
            with col2:
                st.dataframe(df_feat)
            st.markdown('##')
            with st.container():
                st.write("❗Cet outil permet d'assister à la prise de décision et doit être utilisé conjointement avec une analyse approfondie réalisée par un professionel.❗")
        
        with tab2:       
            fig1 = position_global_distrib(df_feat, option)
            st.pyplot(fig1)

        with tab3:
            fig2 = position_label_distrib(df_feat, option)
            st.pyplot(fig2)


 # from terminal : streamlit run app.py
