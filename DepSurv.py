# ============================================================
# BLOC DeepSurv prêt à intégrer dans ton application Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import torch
from pycox.models import DeepSurv
from pycox.evaluation import EvalSurv
import torchtuples as tt
import matplotlib.pyplot as plt

# from PIL import Image
donnee_entre = []
def DeepSurv(donnee_entre):
    # Importation du modèle et des données de test
    chargement_modele = joblib.load("DeepSurv.pkl")
    X_test = joblib.load("X_test_DeepS.pkl")
    Y_test = joblib.load("Y_test_DeepS.pkl")

     # === Prédiction et courbe de survie ===
    x_test = X_test.drop(columns=['Tempsdesuivi (Mois)'], errors='ignore').values.astype('float32')
    surv_df = chargement_modele.predict_surv_df(x_test)

    # Courbe de survie pour le patient sélectionné
    fig, ax = plt.subplots(figsize=(8,5))
    surv_df.iloc[:, 0].plot(ax=ax)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Probabilité de survie")
    ax.set_title("Courbe de survie prédite par DeepSurv")
    st.pyplot(fig)

    # === Évaluation du modèle ===
    try:
        ev = EvalSurv(surv_df, Y_test['Tempsdesuivi (Mois)'], Y_test.astype(bool), censor_surv='km')
        c_index = ev.concordance_td('antolini')
        st.success(f"Concordance Index (C-index) : {c_index:.3f}")
    except Exception as e:
        st.warning(f"Évaluation non disponible : {e}")
