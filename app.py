import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

# title and description
st.title("Prédiction de la capacité de remboursement d'un demandeur de prêt")
st.markdown("Outil de prédiction développé dans le cadre du projet 7 du parcours OC Data Science")

# feature sliders (à voir en fonction de l'analyse feature importance)


# prediction button
if st.button("Prediction"):
    predict(data)

# from terminal : streamlit run app.py
