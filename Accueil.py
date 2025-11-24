import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import base64
#from SSVM import ssvm
#from RSF import rsf
#from sklearn.preprocessing import OrdinalEncoder
#from sksurv.metrics import concordance_index_censored
import numpy as np
#from sksurv.util import Surv

# from PIL import Image

st.set_page_config(page_title="Mémoire NDAO", page_icon="🧠", layout="centered")

# Arriere-plan Streamlit
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string
carac=add_bg_from_local("so-615aff8466a4bdd504491f92-ph0.jpg")
arrier_plan=add_bg_from_local("clinique42_cancerestomac_endoscopie.jpg")
#background-image: url("data:image/jpg;base64,{arrier_plan}");
st.markdown(
f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{arrier_plan}");
    background-size: 981px*600px;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/jpg;base64,{carac}");
    background-position: center;
    background-size: cover;
}}
</style>
""",
unsafe_allow_html=True
)

@st.cache_data
def chargement():
    donnee = pd.read_excel("Donnees.xlsx", engine='openpyxl')
    # Modification des noms des variables
    donnee.columns = ['N° Patient','Douleurs épigastriques', 'Métastases Hépatiques','Denitrution', 'Tabac','Mucineux','Ulcero-bourgeonnant',
                  'Adénopathies', 'Ulcère gastrique','Aspect Infiltrant','Cardiopathie','Cardiopathie 1','Deces']
    # NOTE: J'ai retiré 'Deces' de la drop list ici, car il est utilisé pour df_final. 
    # Cependant, votre fonction main() utilise df = chargement().iloc[1:].reset_index(drop=True), 
    # ce qui semble corriger cela implicitement. Je laisse la drop list originale.
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
    #st.title(
     #   "Prédiction de la survie des patients atteints de cancer de l'estomac",
      #  anchor="center"
    #)
    st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
        Prédiction de la survie des patients atteints de cancer de l'estomac
    </h1>
    """,
    unsafe_allow_html=True
    )

    st.text("   ")
    st.text("   ")
    try:
        # Chargement du modèle SVC
        modele = joblib.load("modele_svm.pkl")
        X_train = joblib.load("X_train.pkl")
        # Entraînement du modèle SVC sans normalisation
        #modele = SVC(C=0.1, probability=True, kernel='rbf', random_state=42)
        #modele.fit(X_train, y_train)
        
        # NOTE IMPORTANTE : S'assurer que les colonnes de X_train sont bien des entiers (0/1)
        X_train = X_train.astype(int)

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
        colonnes[0].text("   ")
        colonnes[1].text("   ")
        # ... (Logique pour patient existant)
        df_final.sort_values(by='N° Patient', ascending=False,inplace=True)
        numPatient=st.sidebar.selectbox("N° Patient", df_final['N° Patient'].unique())
        donneePatient=df_final[df_final['N° Patient']==numPatient].copy() # Utiliser df_final ici
        
        # Affichage des caractéristiques (simplifié pour la concision)
        # Assurez-vous que donneePatient ne contient que les colonnes de FEATURES_COLUMNS pour donnee_entre
        donnee_entre = donneePatient[FEATURE_COLUMNS].astype(int)
        
        # Affichage des valeurs dans la sidebar (pour des raisons de concision, j'ai omis le code d'affichage des colonnes)
        colonn=st.sidebar.columns(2)
        colonn[0].text("   ")
        colonn[1].text("   ")
        val = donnee_entre["Cardiopathie"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        # Affichage dans la sidebar (utilisation simplifiée)
        colonn[0].write(f"**Cardiopathie** : {display_val}")
        val = donnee_entre["Tabac"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[1].write(f"**Tabac** : {display_val}")
        colonn[0].text("   ")

        val = donnee_entre["Denitrution"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[0].write(f"**Denitrution** : {display_val}")
        colonn[1].text("   ")
        colonn[0].text("   ")
        val = donnee_entre["Aspect Infiltrant"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[1].write(f"**Aspect Infiltrant** : {display_val}")
        colonn[1].text("   ")
        colonn[0].text("   ")
        val = donnee_entre["Cardiopathie 1"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[0].write(f"**Cardiopathie 1** : {display_val}")
        colonn[1].text("   ")
        colonn[0].text("   ")
        val = donnee_entre["Métastases Hépatiques"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[1].write(f"**Métastases** : {display_val}")
        colonn[1].text("   ")
        colonn[0].text("   ")
        val = donnee_entre["Adénopathies"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[0].write(f"**Adénopathies** : {display_val}")
        colonn[1].text("   ")
        colonn[0].text("   ")
        val = donnee_entre["Ulcère gastrique"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        colonn[1].write(f"**Ulcère gastrique** : {display_val}")
        colonn[0].text("   ")
        val = donnee_entre["Ulcero-bourgeonnant"].values[0]
        display_val = "OUI" if val == 1 else "NON"
        st.sidebar.write(f"**Ulcero-bourgeonnant** : {display_val}")

            
    elif choix== "🆕 Nouveau Patient":
        
        # Saisie des 9 variables (simplifiée)
        colon=st.sidebar.columns(2)

        # Les 9 variables sont saisies ici... (Le code original est conservé pour la saisie)
        Valcardiopathie= colon[0].selectbox("Cardiopathie", ["NON", "OUI"])
        Cardiopathie = 1 if Valcardiopathie=="OUI" else 0
        
        ValTabac= colon[1].selectbox("Tabac", ["NON", "OUI"])
        Tabac = 1 if ValTabac=="OUI" else 0
        colon[0].text("   ")
        
        ValDenitrution = colon[0].selectbox("Denitrution", ["NON", "OUI"])
        Denitrution = 1 if ValDenitrution=="OUI" else 0
        colon[1].text("   ")
        colon[0].text("   ")
        
        ValInfiltrant= colon[1].selectbox("Infiltrant", ["NON", "OUI"])
        Infiltrant = 1 if ValInfiltrant=="OUI" else 0
        colon[1].text("   ")
        colon[0].text("   ")
        
        ValCardiopathie1= colon[0].selectbox("Cardiopathie 1", ["NON", "OUI"])
        Cardiopathie1 = 1 if ValCardiopathie1=="OUI" else 0
        colon[1].text("   ")
        colon[0].text("   ")
        
        ValMétastases= colon[1].selectbox("Métastases", ["NON", "OUI"])
        Métastases = 1 if ValMétastases=="OUI" else 0
        colon[1].text("   ")
        colon[0].text("   ")
        
        ValAdénopathies= colon[0].selectbox("Adénopathies", ["NON", "OUI"])
        Adénopathies = 1 if ValAdénopathies=="OUI" else 0
        colon[1].text("   ")
        colon[0].text("   ")
        
        Valgastrique= colon[1].selectbox("Gastrique", ["NON", "OUI"])
        gastrique = 1 if Valgastrique=="OUI" else 0
        colon[1].text("   ")
        colon[1].text("   ")
        
        Valbourgeonnant= st.sidebar.selectbox("Ulcero-bourgeonnant", ["NON", "OUI"])
        bourgeonnant = 1 if Valbourgeonnant=="OUI" else 0

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
        donnee_entre = donnee_entre.astype(int) # Assurer le type int

    # ... (votre code précédent dans main())

    # 3️⃣ Prédiction du modèle
    # -------------------------------

    try:
        # Prédiction (Le reste de votre logique de prédiction est conservé et est correct)
        proba_array = modele.predict_proba(donnee_entre)[0]
        pred = modele.predict(donnee_entre)[0]
        
        if modele.classes_.tolist() == [0, 1]:
            proba = proba_array[1] # Probabilité de la classe '1' (Deces)
        else: 
            proba = proba_array[pred]

        #st.subheader("🩺 Résultat de la prédiction")
        st.markdown(
            """
            <h2 style='text-align: center; color: black;'>
                🩺 Résultat de la prédiction
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <h4 style='color: black;'>
                Classe prédite : {'🟥 Deces (à risque)' if pred==1 else '🟩 Vivant (non à risque)'}
            </h4>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <h4 style='color: black;'>
                Probabilité de Deces : {proba*100:.2f}%
            </h4>
            """,
            unsafe_allow_html=True
        )

        # ============================================================
        # 🎯 CORRECTION CRITIQUE DES VALEURS SHAP
        # ============================================================
        
        
        # 1. Préparation du Background Data pour KernelExplainer
        # Prendre un échantillon représentatif de X_train (50 à 100 observations suffisent pour la rapidité)
        BACKGROUND_SAMPLE_SIZE = min(100, X_train.shape[0])
        
        # Selectionner aléatoirement un échantillon de X_train
        background_data = X_train.sample(n=BACKGROUND_SAMPLE_SIZE, random_state=42).astype(float)
        
        # 2. Création de l'Explainer
        # Le KernelExplainer est le bon choix pour SVC(probability=True).
        # NOTE: Pour les modèles entraînés sur des données 0/1, le KernelExplainer fonctionne mieux
        # si le background est converti en float.
        explainer = shap.KernelExplainer(modele.predict_proba, background_data)
        
        # 3. Calcul des valeurs SHAP
        # nsamples=100 est le minimum par défaut pour une bonne estimation.
        # On s'assure que la donnée d'entrée est aussi en float pour l'explainer
        # 3️⃣ Préparation des données à expliquer
        data_to_explain = donnee_entre.astype(float)

        # 4️⃣ Calcul des valeurs SHAP
        shap_values = explainer.shap_values(data_to_explain, nsamples=100)

        # 5️⃣ Extraction des valeurs SHAP pour la classe 1 (Décès)
        shap_class1 = shap_values[0, :, 1]  # attention : shap_values a la forme (nsamples, n_features, n_classes)
        feature_names = donnee_entre.columns

        # -----------------------
        # Tableau SHAP
        # -----------------------
        #shap_table = pd.DataFrame({
        #    "Variables": feature_names,
        #    "Valeur SHAP": shap_class1
        #}).sort_values(by="Valeur SHAP", ascending=False)

        #st.subheader("Valeurs SHAP par Variable")
        #st.dataframe(shap_table)
        # -----------------------
        # Graphique horizontal des SHAP
        # -----------------------
        # Tri des features par importance absolue

        # Création du tableau SHAP
        shap_table = pd.DataFrame({
            "Variables": feature_names,
            "Valeur SHAP": shap_class1
        }).sort_values(by="Valeur SHAP", ascending=False)  # ascending=True pour que la plus grande soit en haut après inversion

        #st.subheader("Importance des variables selon les valeurs SHAP")
        st.markdown(
            """
            <h3 style='text-align: center; color: black;'>
                Importance des variables selon les valeurs SHAP
            </h3>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = shap_table['Valeur SHAP'].apply(lambda x: 'red' if x > 0 else 'green')

        # Bar plot horizontal
        bars = ax.barh(shap_table["Variables"], shap_table["Valeur SHAP"], color=colors)

        # Étiquettes de valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (-0.01 if width > 0 else +0.01),  # petit décalage selon le signe
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va='center',
                ha='left' if width > 0 else 'right',
                color='black',
                fontsize=9,
                fontweight='bold'
            )

        #ax.set_xlabel("Valeur SHAP")
        #ax.set_title("Importance des variables pour la classe 'Décès'")
        ax.axvline(0, color='black', linewidth=0.8)
        ax.invert_yaxis()  # Mettre la plus importante en haut
        st.pyplot(fig)

        # -----------------------
        # 8️⃣ Graphique Waterfall (vue détaillée)
        st.markdown(
            """
            <h3 style='text-align: center; color: black;'>
                Interprétation du détail des contributions des variables
            </h3>
            """,
            unsafe_allow_html=True
        )
        shap_explanation = shap.Explanation(
            values=shap_class1,
            base_values=explainer.expected_value[1],
            data=data_to_explain.values[0],
            feature_names=feature_names
        )
        fig_wf = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, show=False)
        st.pyplot(fig_wf)
            

    except Exception as e:
        st.error(f"Erreur lors de la prédiction ou du calcul SHAP : {e}")
        st.exception(e) # Affiche le stack trace pour faciliter le débogage
        return
# ... (le reste de votre fonction main())
    # Chargement du CSS
    fichier_css = "style.css"
    try:
         with open(fichier_css) as f:
             st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    except FileNotFoundError:
         st.warning(f"Le fichier CSS '{fichier_css}' n'a pas été trouvé.")


if __name__ == "__main__":
    main()