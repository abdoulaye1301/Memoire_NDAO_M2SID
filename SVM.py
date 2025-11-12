import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ===========================================
# ⚙️ Fonction principale
# ===========================================
def svm_app(df_patients):
    # --------------------------------------------------------
    # 1️⃣ Chargement du modèle et de la base patients
    # --------------------------------------------------------
    try:
        model = joblib.load("modele_svm.pkl")
        X_test = joblib.load("X_test.pkl")
        st.success("✅ Modèle SVM chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return

    # --------------------------------------------------------
    # 2️⃣ Choix du mode : nouveau patient ou existant
    # --------------------------------------------------------
    choix = st.radio("Sélectionnez une option :", ["🆕 Ajouter un nouveau patient", "📂 Sélectionner un patient existant"])

    if choix == "📂 Sélectionner un patient existant" and not df_patients.empty:
        patient_id = st.selectbox("Choisir un patient :", df_patients.index)
        donnee_entre = df_patients.loc[[patient_id]]
        st.dataframe(donnee_entre)
    else:
        st.subheader("🧩 Saisir les informations du nouveau patient")

        # Exemple avec 3 variables (à adapter selon ton modèle)
        age = st.number_input("Âge", 0, 120, 50)
        cholesterol = st.number_input("Cholestérol", 100, 400, 200)
        tabac = st.selectbox("Fumeur ?", [0, 1])

        donnee_entre = pd.DataFrame({
            'age': [age],
            'cholesterol': [cholesterol],
            'tabac': [tabac]
        })

        if st.button("💾 Ajouter à la base"):
            df_patients = pd.concat([df_patients, donnee_entre], ignore_index=True)
            df_patients.to_csv("patients.csv", index=False)
            st.success("✅ Nouveau patient ajouté !")

    # --------------------------------------------------------
    # 3️⃣ Prédiction du modèle
    # --------------------------------------------------------
    if st.button("🔍 Prédire avec le modèle SVM"):
        try:
            proba = model.predict_proba(donnee_entre)[0][1]

            pred = model.predict(donnee_entre)[0]
            st.subheader("🩺 Résultat de la prédiction")
            st.write(f"**Classe prédite :** {'🟥 Positif (à risque)' if pred==1 else '🟩 Négatif (non à risque)'}")
            st.write(f"**Probabilité de risque :** {proba:.2f}")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return

        # --------------------------------------------------------
        # 4️⃣ Valeurs SHAP pour interprétation
        # --------------------------------------------------------
        st.subheader("📊 Interprétation SHAP du modèle SVM")

        try:
            # Sélection d’un échantillon de fond pour l’explication
            #background = shap.sample(X_test.dropna(), 50)
            background = shap.sample(X_test, 50, random_state=42)


            # Création de l’explainer SHAP (KernelExplainer pour SVM)
            explainer = shap.KernelExplainer(model.predict_proba, background)

            # Calcul des valeurs SHAP pour ce patient
            shap_values = explainer.shap_values(donnee_entre)
            

            # Gestion automatique selon le type de sortie
            if isinstance(shap_values, list):
                # Cas d'une liste (ex: shap_values = [classe0, classe1])
                shap_array = shap_values[1][0]   # Dernière classe = "classe positive"
            else:
                # Cas d'un seul tableau (shape = (1, n_features))
                shap_array = shap_values[0]

            # Création du DataFrame des valeurs SHAP
            shap_df = pd.DataFrame({
                "Variable": donnee_entre.columns,
                "Valeur_SHAP": shap_array,
                "Valeur_patient": donnee_entre.values[0],
            }).sort_values("Valeur_SHAP", ascending=False)

            shap_df["Effet"] = shap_df["Valeur_SHAP"].apply(
                lambda x: "⬆️ Augmente le risque" if x > 0 else "⬇️ Diminue le risque"
            )

            st.dataframe(shap_df, use_container_width=True)

            # Graphique barres horizontales
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = shap_df["Valeur_SHAP"].apply(lambda x: "red" if x > 0 else "green")
            ax.barh(shap_df["Variable"], shap_df["Valeur_SHAP"], color=colors)
            ax.set_xlabel("Valeur SHAP (impact sur la prédiction)")
            ax.set_title("Impact des variables (SHAP)")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # Importance globale
            st.subheader("🌍 Importance globale des variables")
            shap.summary_plot(shap_values[-1] if isinstance(shap_values, list) else shap_values,
                            X_test, show=False)
            st.pyplot(bbox_inches="tight")

        except Exception as e:
            st.error(f"Erreur lors du calcul des valeurs SHAP : {e}")


# ===========================================
# 🔹 Lancement de l’app
# ===========================================
if __name__ == "__main__":
    svm_app()
