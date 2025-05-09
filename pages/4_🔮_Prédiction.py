import streamlit as st
import pandas as pd
import dill
import matplotlib.pyplot as plt
from utils import apply_feature_engineering_streamlit # Assurez-vous que les listes sont bien gérées ici
from loguru import logger

st.set_page_config(page_title="Prédiction ", page_icon="🔮", layout="wide")
st.markdown("# Prédiction de Fraude sur Nouvelles Données")

model = st.session_state.get('model')
raw_cols_for_pipeline = st.session_state.get('raw_training_cols_for_pipeline') # Colonnes après FE manuel, avant preprocessor
products_list = st.session_state.get('products_list_recodage', []) # Récupérer les listes
provider_list = st.session_state.get('provider_list_recodage', [])

if model is None or raw_cols_for_pipeline is None:
    st.error("Modèle ou liste de colonnes non chargés. Prédiction impossible.")
    st.stop()

uploaded_file = st.file_uploader("Chargez CSV pour prédiction", type="csv", key="pred_uploader")

if uploaded_file is not None:
    try:
        df_new_raw = pd.read_csv(uploaded_file, sep=",")
        st.info(f"Fichier '{uploaded_file.name}' chargé. {df_new_raw.shape[0]} lignes.")
        st.dataframe(df_new_raw.head(3))

        with st.spinner("Application du Feature Engineering..."):
            # Appliquer FE. Les listes de recodage sont récupérées de session_state (calculées dans main_app)
            df_new_featured = apply_feature_engineering_streamlit(df_new_raw.copy(), products_list, provider_list)
            logger.info(f"Feature engineering appliqué sur nouvelles données. Shape: {df_new_featured.shape}")
            logger.debug(f"Colonnes après FE sur nouvelles données: {df_new_featured.columns.tolist()}")


        # S'assurer que toutes les colonnes attendues par le pipeline (après FE manuel) sont présentes
        # et dans le bon ordre.
        df_for_pipeline_pred = pd.DataFrame(columns=raw_cols_for_pipeline) # Créer un DF vide avec le bon ordre
        for col in raw_cols_for_pipeline:
            if col in df_new_featured.columns:
                df_for_pipeline_pred[col] = df_new_featured[col]
            else:
                logger.warning(f"Colonne attendue '{col}' manquante après FE sur nouvelles données. Remplie avec 0.")
                df_for_pipeline_pred[col] = 0 # Ou np.nan, puis gérer par le pipeline si possible

        st.subheader("Données prêtes pour le pipeline (après FE et alignement des colonnes)")
        st.dataframe(df_for_pipeline_pred.head(3))


        with st.spinner("Prédiction en cours..."):
            predictions = model.predict(df_for_pipeline_pred)
            predictions_proba = model.predict_proba(df_for_pipeline_pred)[:, 1]

        results_df = pd.DataFrame({'Prediction_Fraude': predictions, 'Probabilite_Fraude': predictions_proba * 100})

        # Ajouter TransactionId si présent dans le fichier uploadé initialement
        if 'TransactionId' in df_new_raw.columns: # Vérifier dans df_new_raw
            results_df = pd.concat([df_new_raw[['TransactionId']].reset_index(drop=True), results_df], axis=1)

        st.subheader("Résultats des Prédictions")
        st.dataframe(results_df)

        # Visualisation
        pred_counts = results_df['Prediction_Fraude'].value_counts()
        fig, ax = plt.subplots()
        pred_counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title('Distribution des Prédictions')
        ax.set_xticklabels(['Non-Fraude (0)', 'Fraude (1)'], rotation=0)
        st.pyplot(fig)
        
        st.write(pred_counts.reset_index().rename(columns={'index':'Prédiction', 'Prediction_Fraude':'Nombre'}))

        csv_res = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger Prédictions", csv_res, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
        st.exception(e)