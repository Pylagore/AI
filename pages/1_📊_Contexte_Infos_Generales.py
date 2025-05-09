import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Contexte & Infos ", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align : center;'>Contexte et Informations Générales</h1>",unsafe_allow_html=True)

st.sidebar.header("Infos Générales")
st.write(
    """Cette page présente un aperçu général du jeu de données d'entraînement,
       y compris les informations sur les types de données, les valeurs manquantes
       et le taux de fraude.L'objectif principal est de développer un modèle
       d'apprentissage automatique capable de détecter avec précision les transactions
       frauduleuses sur des plateformes de commerce électronique."""
)

#if 'df_processed_for_eda' not in st.session_state or st.session_state['df_processed_for_eda'] is None:
   # st.error("Les données traitées pour EDA ne sont pas disponibles. Retournez à la page principale.")
    #st.stop()

df_eda = st.session_state['df_processed_for_eda'] # DataFrame après tout le FE manuel
df_raw = st.session_state.get('df_initial_load') # DataFrame original chargé

#st.subheader("Informations sur les Données Brutes (avant FE)")
#if df_raw is not None:
 #   st.markdown(f"**Dimensions Brutes:** {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")
  #  buffer_raw = StringIO()
   # df_raw.info(buf=buffer_raw)
   # st.text(buffer_raw.getvalue())
   # st.markdown("**Valeurs Manquantes Brutes:**")
   # st.dataframe(df_raw.isnull().sum().rename("Nb Manquants").to_frame().T)
   # st.markdown(f"**Doublons Bruts:** {df_raw.duplicated().sum()}")
#else:
  #  st.warning("Données brutes non disponibles.")

st.markdown("---")
st.markdown("<h3 style='text-align : center;'>Informations sur les Données</h3>",unsafe_allow_html=True)
st.markdown(f"**Dimensions Traitées :** {df_eda.shape[0]} lignes, {df_eda.shape[1]} colonnes")

#st.markdown("### Types de données (EDA)")
#buffer_eda = StringIO()
#df_eda.info(buf=buffer_eda)
#st.text(buffer_eda.getvalue())

st.markdown("<h3 style='text-align : center;'>Valeurs Manquantes</h3>",unsafe_allow_html=True)
st.dataframe(df_eda.isnull().sum().rename("Nb Manquants").to_frame().T)

#st.markdown(f"### Doublons (EDA): {df_eda.duplicated().sum()}")

st.markdown("<h3 style='text-align : center;'>Statistiques Descriptives</h3>",unsafe_allow_html=True)
st.dataframe(df_eda.describe(include='all'))

st.markdown("<h3 style='text-align : center;'>Taux de Fraude</h3>",unsafe_allow_html=True)
if 'FraudResult' in df_eda.columns:
    fraud_counts = df_eda["FraudResult"].value_counts()
    fraud_percentage_series = df_eda["FraudResult"].value_counts(normalize=True) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Non Frauduleuses (0)", f"{fraud_counts.get(0, 0)}")
        st.metric("Frauduleuses (1)", f"{fraud_counts.get(1, 0)}")
    with col2:
        st.metric("% Non Frauduleuses", f"{fraud_percentage_series.get(0, 0):.2f}%")
        st.metric("% Frauduleuses", f"{fraud_percentage_series.get(1, 0):.2f}%")

    labels = ['Non-Fraude (0)', 'Fraude (1)']
    sizes = [fraud_percentage_series.get(0,0), fraud_percentage_series.get(1,0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90, colors=['skyblue', 'orange'], pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title('Répartition Fraude / Non-Fraude')
    st.pyplot(fig)
else:
    st.warning("Colonne 'FraudResult' non trouvée dans les données.")