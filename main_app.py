import streamlit as st
import pandas as pd
import numpy as np
import dill
from pathlib import Path
from utils import apply_feature_engineering_streamlit
from loguru import logger

st.set_page_config(
    page_title="Détection Fraude", page_icon="🇸🇳", layout="wide", initial_sidebar_state="expanded"
)

@st.cache_data
def load_initial_data(file_path, sep):
    try:
        df = pd.read_csv(file_path, sep=",")
        if 'PricingStrategy' in df.columns: # S'assurer du type pour la suite
            df['PricingStrategy'] = df['PricingStrategy'].astype('str')
        return df
    except FileNotFoundError: st.error(f"Fichier '{file_path}' introuvable."); return None
    except Exception as e: st.error(f"Erreur chargement CSV: {e}"); return None

@st.cache_resource
def load_model_and_artifacts(_model_path, _raw_cols_path): # _ pour éviter conflit de nom avec variable globale
    try:
        with open(_model_path, 'rb') as f: model_loaded = dill.load(f)
        with open(_raw_cols_path, 'rb') as f: raw_cols_loaded = dill.load(f)
        return model_loaded, raw_cols_loaded
    except FileNotFoundError: st.error(f"Fichier modèle ou colonnes introuvable."); return None, None
    except Exception as e: st.error(f"Erreur chargement modèle/colonnes: {e}"); return None, None

# --- Initialisation ---
DATA_FILE_PATH = "training.csv"
MODEL_FILE_PATH = 'best_Regression_logistique.dill'
RAW_COLS_FILE_PATH = 'training_raw_features.dill' # Colonnes après FE manuel, attendues par le preprocessor


if 'products_list_recodage' not in st.session_state:
    df_temp_for_lists = load_initial_data(DATA_FILE_PATH, sep=",")
    if df_temp_for_lists is not None and 'ProductId' in df_temp_for_lists.columns:
        frequencies_product = df_temp_for_lists['ProductId'].value_counts(normalize=True) * 100
        products_above_12 = frequencies_product[frequencies_product > 12]
        st.session_state['products_list_recodage'] = products_above_12.index.tolist()
        logger.info(f"Liste ProductId pour recodage (Streamlit): {st.session_state['products_list_recodage']}")
    else:
        st.session_state['products_list_recodage'] = [] # Fallback

    if df_temp_for_lists is not None and 'ProviderId' in df_temp_for_lists.columns:
        frequencies_provider = df_temp_for_lists['ProviderId'].value_counts(normalize=True) * 100
        provider_above_12 = frequencies_provider[frequencies_provider > 12]
        st.session_state['provider_list_recodage'] = provider_above_12.index.tolist()
        logger.info(f"Liste ProviderId pour recodage (Streamlit): {st.session_state['provider_list_recodage']}")
    else:
        st.session_state['provider_list_recodage'] = [] # Fallback
    del df_temp_for_lists


if 'df_initial_load' not in st.session_state:
    st.session_state['df_initial_load'] = load_initial_data(DATA_FILE_PATH, sep=";")

if 'df_processed_for_eda' not in st.session_state and st.session_state['df_initial_load'] is not None:
    logger.info("Application du feature engineering pour l'EDA...")
    st.session_state['df_processed_for_eda'] = apply_feature_engineering_streamlit(
        st.session_state['df_initial_load'].copy(), # Utiliser une copie pour le FE
        st.session_state['products_list_recodage'],
        st.session_state['provider_list_recodage']
    )
    if st.session_state['df_processed_for_eda'] is not None:
         logger.info(f"Données pour EDA traitées. Shape: {st.session_state['df_processed_for_eda'].shape}")
         logger.debug(f"Colonnes df_processed_for_eda: {st.session_state['df_processed_for_eda'].columns.tolist()}")


if 'model' not in st.session_state: # Charger une seule fois
    model, raw_cols = load_model_and_artifacts(MODEL_FILE_PATH, RAW_COLS_FILE_PATH)
    st.session_state['model'] = model
    st.session_state['raw_training_cols_for_pipeline'] = raw_cols # Colonnes attendues par le preprocessor

# --- Affichage ---
st.markdown("<h1 style='text-align : center;'>🇸🇳 Application de Détection de Fraude</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align : center;'>Analyse exploratoire et prédictions de fraude.</h3>",unsafe_allow_html=True)
col1,col2,col3 = st.columns([1,2,1])
with col2:
    gif_url = "https://www.gif-maniac.com/gifs/50/49800.gif"
    st.image(gif_url, caption="Sénégal", width=300)

st.sidebar.success("Navigation")

if st.session_state.get('df_initial_load') is None:
    st.error("Données initiales non chargées. Vérifiez training.csv.")
    st.stop()
if st.session_state.get('df_processed_for_eda') is None:
    st.error("Erreur de feature engineering pour EDA.")
    st.stop()
if st.session_state.get('model') is None:
    st.warning("Modèle non chargé. La prédiction ne fonctionnera pas.")

st.markdown("<h3 style='text-align : center;'>Aperçu des données brutes initiales.</h3>",unsafe_allow_html=True)
st.dataframe(st.session_state['df_initial_load'].head())
st.markdown("<h3 style='text-align : center;'>Aperçu des données après feature engineering (pour EDA).</h3>",unsafe_allow_html=True)
st.dataframe(st.session_state['df_processed_for_eda'].head())

if st.session_state.get('raw_training_cols_for_pipeline'):
    st.caption(f"Le modèle attend {len(st.session_state['raw_training_cols_for_pipeline'])} colonnes (après feature engineering manuel) : {st.session_state['raw_training_cols_for_pipeline'][:5]}...")