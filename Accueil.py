import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
#from SSVM import ssvm
#from RSF import rsf
#from sklearn.preprocessing import OrdinalEncoder
#from sksurv.metrics import concordance_index_censored
#import numpy as np
#from sksurv.util import Surv

# from PIL import Image

st.set_page_config(page_title="Mémoire NDAO", page_icon="🧠", layout="centered")
@st.cache_data
def chargement():
    donnee = pd.read_excel("Donnees.xlsx", engine='openpyxl')
    # Modification des noms des variables
    donnee.columns = ['N° Patient','Douleurs épigastriques', 'Métastases Hépatiques','Denitrution', 'Tabac','Mucineux','Ulcero-bourgeonnant',
                 'Adénopathies', 'Ulcère gastrique','Aspect Infiltrant','Cardiopathie','Cardiopathie 1','Deces']
    donnee.drop(["Douleurs épigastriques","Mucineux",'Deces'],axis=1,inplace=True)
    return donnee

# Definition de la fonction principale
def main():
    # ⚠️ Définir la liste des caractéristiques
    # ------------------------------------------------------------
    FEATURE_COLUMNS = [
        'Métastases Hépatiques', 'Denitrution', 'Tabac', 'Ulcero-bourgeonnant', 
        'Adénopathies', 'Ulcère gastrique', 'Aspect Infiltrant', 
        'Cardiopathie', 'Cardiopathie 1'
    ]
    st.title(
        "Prédiction de la survie des patients atteints de cancer de l'estomac"
    )
    st.text("   ")
    st.text("   ")
    try:
        model = joblib.load("modele_svm.pkl")
        X_test = joblib.load("X_test.pkl")
        X_train = joblib.load("X_train.pkl")
        #st.success("✅ Modèle SVM chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return

    df = chargement().iloc[1:].reset_index(drop=True)
    # Collecte des données du patient

    
    df_final= df.copy()
    
     # 2️⃣ Choix du mode : nouveau patient ou existant
    # --------------------------------------------------------
    choix = st.sidebar.selectbox("**NAVIGATION** :", ["🆕 Nouveau Patient", "📂 Patient existant"])
    colonnes=st.sidebar.columns(2)
    colonnes[1].subheader("🧩  **du patient**")
    colonnes[0].subheader("**Caractéristiques**")
    if choix== "📂 Patient existant":

        df_final.sort_values(by='N° Patient', ascending=False,inplace=True)
        numPatient=st.sidebar.selectbox("N° Patient", df_final['N° Patient'].unique())
        donneePatient=df[df_final['N° Patient']==numPatient]
    
    #==================================================================#
       # choix=st.selectbox("Navigation", ["RSF"], key="navigation")
        #if choix=="RSF":
        colon=st.sidebar.columns(2)
        #Ulcere_gastrique = colon[1].selectbox("Ulcere Gastrique", ("NON", "OUI"))
        Valcardiopathie=donneePatient['Cardiopathie'].values[0]
        if Valcardiopathie==1:
            colon[0].write(f"**Cardiopathie** : OUI")
        else:
            colon[0].write(f"**Cardiopathie** : NON")
        #colon[0].write(f"**Cardiopathie** : {donneePatient['Cardiopathie'].values[0]}")
        ValTabac=donneePatient['Tabac'].values[0]
        if ValTabac==1:
            colon[1].write(f"**Tabac** : OUI")
        else:
            colon[1].write(f"**Tabac** : NON")
        #colon[1].write(f"**Tabac :** {donneePatient['Tabac'].values[0]}")
        colon[0].text("   ")
        #Constipation = colon[1].selectbox("Constipation", ("NON", "OUI"))
        #Denitrution = colon[0].selectbox("Denitrution", ("NON", "OUI"))
        ValDenitrution=donneePatient['Denitrution'].values[0]
        if ValDenitrution==1:
            colon[0].write(f"**Denitrution** : OUI")
        else:
            colon[0].write(f"**Denitrution** : NON")
        #colon[0].write(f"**Denitrution :** {donneePatient['Denitrution'].values[0]}")
        #Tubuleux = colon[0].selectbox("Tubuleux", ("NON", "OUI"))
        colon[1].text("   ")
        colon[0].text("   ")
        ValInfiltrant=donneePatient['Aspect Infiltrant'].values[0]
        if ValInfiltrant==1:
            colon[1].write(f"**Infiltrant** : OUI")
        else:
            colon[1].write(f"**Infiltrant** : NON")
        #colon[1].write(f"**Infiltrant :** {donneePatient['Aspect Infiltrant'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValCardiopathie1=donneePatient['Cardiopathie 1'].values[0]
        if ValCardiopathie1==1:
            colon[0].write(f"**Cardiopathie 1** : OUI")
        else:
            colon[0].write(f"**Cardiopathie 1** : NON")
        #colon[0].write(f"**Cardiopathie 1 :** {donneePatient['Cardiopathie 1'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValMétastases=donneePatient['Métastases Hépatiques'].values[0]
        if ValMétastases==1:
            colon[1].write(f"**Métastases** : OUI")
        else:
            colon[1].write(f"**Métastases** : NON")
        #colon[1].write(f"**Metastases :** {donneePatient['Métastases Hépatiques'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValAdénopathies=donneePatient['Adénopathies'].values[0]
        if ValAdénopathies==1:
            colon[0].write(f"**Adénopathie** : OUI")
        else:
            colon[0].write(f"**Adénopathie** : NON")
        #colon[0].write(f"**Adénopathie :** {donneePatient['Adénopathies'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        Valgastrique=donneePatient['Ulcère gastrique'].values[0]
        if Valgastrique==1:
            colon[1].write(f"**Gastrique** : OUI")
        else:
            colon[1].write(f"**Gastrique** : NON")
        #colon[1].write(f"**Ulcère Gastrique :** {donneePatient['Ulcère gastrique'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        Valgastrique=donneePatient['Ulcero-bourgeonnant'].values[0]
        if Valgastrique==1:
            st.sidebar.write(f"**Ulcero-bourgeonnant** : OUI")
        else:
            st.sidebar.write(f"**Ulcero-bourgeonnant** : NON")
        #st.sidebar.write(f"**Ulcero-bourgeonnant :** {donneePatient['Ulcero-bourgeonnant'].values[0]}")




        #donne2 = patient()
        #donnee_entre = pd.concat([donne2,df], axis=0)
        donnee_entre = donneePatient.drop(columns=['N° Patient'])

        donnee_entre = donnee_entre.astype(int)

        # Récupération de la première ligne (nouveau patient)
        donnee_entre = donnee_entre[:1]
       # rsf(donnee_entre)
    elif choix== "🆕 Nouveau Patient":
        
        # Saisie des 9 variables (simplifiée)
        # Assurez-vous que l'ordre des variables ici correspond à celui de FEATURE_COLUMNS
        
        
        
        colon=st.sidebar.columns(2)
        # ... autres saisies pour les 9 features ...


        Valcardiopathie= colon[0].selectbox("Cardiopathie", ["NON", "OUI"])
        if Valcardiopathie=="OUI":
            Cardiopathie=1
        else:
            Cardiopathie=0
        #colon[0].write(f"**Cardiopathie** : {donneePatient['Cardiopathie'].values[0]}")
        
        ValTabac= colon[1].selectbox("Tabac", ["NON", "OUI"])
        if ValTabac=="OUI":
            Tabac=1
        else:
            Tabac=0
        #colon[1].write(f"**Tabac :** {donneePatient['Tabac'].values[0]}")
        colon[0].text("   ")
        #Constipation = colon[1].selectbox("Constipation", ("NON", "OUI"))
        #Denitrution = colon[0].selectbox("Denitrution", ("NON", "OUI"))
        ValDenitrution = colon[0].selectbox("Denitrution", ["NON", "OUI"])
        if ValDenitrution=="OUI":
            Denitrution=1
        else:
            Denitrution=0
        #colon[0].write(f"**Denitrution :** {donneePatient['Denitrution'].values[0]}")
        #Tubuleux = colon[0].selectbox("Tubuleux", ("NON", "OUI"))
        colon[1].text("   ")
        colon[0].text("   ")
        ValInfiltrant= colon[1].selectbox("Infiltrant", ["NON", "OUI"])
        if ValInfiltrant=="OUI":
            Infiltrant=1
        else:
            Infiltrant=0
        #colon[1].write(f"**Infiltrant :** {donneePatient['Aspect Infiltrant'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValCardiopathie1= colon[0].selectbox("Cardiopathie 1", ["NON", "OUI"])
        if ValCardiopathie1=="OUI":
            Cardiopathie1=1
        else:
            Cardiopathie1=0
        #colon[0].write(f"**Cardiopathie 1 :** {donneePatient['Cardiopathie 1'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValMétastases= colon[1].selectbox("Métastases", ["NON", "OUI"])
        if ValMétastases=="OUI":
            Métastases=1
        else:
            Métastases=0
        #colon[1].write(f"**Metastases :** {donneePatient['Métastases Hépatiques'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        ValAdénopathies= colon[0].selectbox("Adénopathies", ["NON", "OUI"])
        if ValAdénopathies=="OUI":
            Adénopathies=1
        else:
            Adénopathies=0
        #colon[0].write(f"**Adénopathie :** {donneePatient['Adénopathies'].values[0]}")
        colon[1].text("   ")
        colon[0].text("   ")
        Valgastrique= colon[1].selectbox("Gastrique", ["NON", "OUI"])
        if Valgastrique=="OUI":
            gastrique=1
        else:
            gastrique=0
        #colon[1].write(f"**Ulcère Gastrique :** {donneePatient['Ulcère gastrique'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        Valbourgeonnant= st.sidebar.selectbox("Ulcero-bourgeonnant", ["NON", "OUI"])
        if Valbourgeonnant=="OUI":
            bourgeonnant=1
        else:
            bourgeonnant=0


        # Créez le DataFrame avec TOUTES les 9 colonnes
        donnee_entre_dict = {
            'Métastases Hépatiques': [Métastases],
            'Denitrution': [Denitrution],
            'Tabac': [Tabac],
            'Ulcero-bourgeonnant': [bourgeonnant],
            'Adénopathies': [Adénopathies],
            'Ulcère gastrique': [gastrique],
            'Aspect Infiltrant': [Infiltrant],
            'Cardiopathie': [Cardiopathie],
            'Cardiopathie 1': [Cardiopathie1]
        }
        
        # Utilisez FEATURE_COLUMNS pour garantir l'ordre des colonnes
        donnee_entre = pd.DataFrame(donnee_entre_dict, columns=FEATURE_COLUMNS)
        
        # Le reste du code de sauvegarde n'est pas pertinent pour l'erreur SHAP
        # ... (le code de sauvegarde du patient)
        # ...

    # 3️⃣ Prédiction du modèle
    # --------------------------------------------------------
    #if st.button("🔍 Prédire avec le modèle SVM"):
    # -------------------------------
    # 3️⃣ Prédiction et valeurs SHAP
    # -------------------------------

    try:
        # Vérification des données du patient
        #st.write("Shape de donnee_entre :", donnee_entre.shape)
        #st.write("Colonnes :", donnee_entre.columns.tolist())
        #st.write("Classes du modèle :", model.classes_)

        # Prédiction
        proba_array = model.predict_proba(donnee_entre)[0]  # tableau [prob_class0, prob_class1]
        pred = model.predict(donnee_entre)[0]

        # Probabilité de la classe '1'
        if len(proba_array) == 2:
            proba = proba_array[1]
        else:
            proba = proba_array[0]  # cas improbable, si une seule classe

        st.subheader("🩺 Résultat de la prédiction")
        st.write(f"**Classe prédite :** {'🟥 Deces (à risque)' if pred==1 else '🟩 Vivant (non à risque)'}")
        st.write(f"**Probabilité de Deces :** {proba:.2f}")

        # ============================================================
       # ... (après l'affichage de la probabilité) ...

        # ============================================================
        # ============================================================
       # ============================================================
        # 🎯 VALEURS SHAP (Calcul Optimisé pour SVM RBF Binaire)
        # ============================================================
        st.subheader("📊 Interprétation du modèle (valeurs SHAP)")

        # --- 1. Préparation des données d'arrière-plan (Background Data) ---
        import numpy as np

        # Préparation du jeu de données complet d'entraînement pour le clustering
        X_train_full_df = df_final.drop(columns=['N° Patient'])[FEATURE_COLUMNS]

        # Paramètre de clustering (plus le nombre est élevé, plus c'est précis mais lent)
        N_CLUSTERS = 100 

        # 1. Calcul des centres K-Means pour l'échantillon de fond
        clustering = shap.kmeans(X_train_full_df, N_CLUSTERS)

        # 2. 🔴 Adaptation Binaire : Projection des centres sur les valeurs 0 ou 1.
        # Cela garantit que les données d'arrière-plan sont cohérentes avec votre modèle binaire.
        X_background_binary = clustering.data.round().astype(int)

        # --- 2. Initialisation de l'Explainer ---

        # La fonction de prédiction cible la probabilité de la CLASSE 1 (risque)
        predictor = lambda x: model.predict_proba(x)[:, 1]

        # 🔴 SOLUTION : Passer le tableau NumPy pur X_background_binary
        # Supprimez l'utilisation de shap.maskers.Independent ici
        explainer = shap.KernelExplainer(predictor, X_background_binary)

        # --- 3. Calcul SHAP pour le patient unique ---
        # Le calcul est effectué sur le DataFrame du patient (donnee_entre)
        # shap_values sera une liste de deux tableaux (un par classe)
        shap_values = explainer.shap_values(donnee_entre) 

        # --- 4. Extraction des valeurs et de la base (Classe 1) ---

        # Valeurs SHAP pour la CLASSE POSITIVE (index 1)
        shap_values_class_1 = shap_values[1] 

        # Valeurs SHAP pour le PATIENT UNIQUE (première et seule ligne)
        shap_values_patient = np.asarray(shap_values_class_1).flatten() 

        # La valeur de base pour la classe 1 (moyenne des probabilités de fond)
        expected_value = explainer.expected_value[1] 

        # --- 5. Diagnostic et Affichage ---
        patient_data = np.asarray(donnee_entre.iloc[0].values).flatten() 
        feature_names = donnee_entre.columns.tolist()

        if shap_values_patient.shape[0] != len(feature_names):
            st.error(f"❌ ERREUR CRITIQUE: La forme de SHAP est {shap_values_patient.shape[0]} mais on attend {len(feature_names)} caractéristiques. Vérifiez l'extraction.")
            return

        # ------------------------------------------------------------
        # Affichage du DataFrame et du Waterfall Plot (pas de changement)
        # ------------------------------------------------------------

        shap_df = pd.DataFrame(shap_values_patient, index=feature_names, columns=["Valeur SHAP"])
        shap_df["Impact"] = shap_df["Valeur SHAP"].apply(lambda x: "⬆️ augmente le risque" if x > 0 else "⬇️ diminue le risque")
        st.dataframe(shap_df.sort_values(by="Valeur SHAP", ascending=False).style.format({"Valeur SHAP": "{:.3f}"}))

        st.write("### 🔍 Graphique SHAP (Détail de la Prédiction)")

        shap_explanation = shap.Explanation(
            values=shap_values_patient,
            base_values=expected_value,
            data=patient_data,
            feature_names=feature_names
        )

        waterfall_fig = shap.plots.waterfall(
            shap_explanation,
            show=False
        )
        st.pyplot(waterfall_fig, bbox_inches='tight')
    except Exception as e:
        st.error(f"Erreur lors de la prédiction ou du calcul SHAP : {e}")
        return
    # Chargement du CSS
    fichier_css = "style.css"
    with open(fichier_css) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

if __name__ == "__main__":
    main()


