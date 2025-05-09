import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Univariée ", page_icon="📈", layout="wide")
st.markdown("<h1 style='text-align : center;'>Analyse Univariée</h1>",unsafe_allow_html=True)

if 'df_processed_for_eda' not in st.session_state or st.session_state['df_processed_for_eda'] is None:
    st.error("Données EDA non disponibles.")
    st.stop()
df_eda = st.session_state['df_processed_for_eda']

# Colonnes après FE manuel, comme définies pour le modèle (sauf la cible pour les features)
raw_cols_for_pipeline = st.session_state.get('raw_training_cols_for_pipeline', [])
if not raw_cols_for_pipeline:
    st.warning("Liste des colonnes pour le pipeline non chargée, l'identification des types peut être imprécise.")
    # Fallback si la liste n'est pas là
    numeric_cols_eda = df_eda.select_dtypes(include=np.number).columns.tolist()
    if 'FraudResult' in numeric_cols_eda: numeric_cols_eda.remove('FraudResult')
    categorical_cols_eda = df_eda.select_dtypes(include='object').columns.tolist()
else:
    # Utiliser les colonnes brutes attendues par le pipeline pour identifier les types, car c'est ce qui est pertinent
    temp_df_for_type_detection = df_eda[raw_cols_for_pipeline] # Créer un df temporaire avec seulement ces colonnes
    numeric_cols_eda = temp_df_for_type_detection.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_eda = temp_df_for_type_detection.select_dtypes(include='object').columns.tolist()


st.sidebar.header("Sélection Variable")
analysis_type = st.sidebar.radio("Type de Variable", ["Catégorielle", "Numérique"])

if analysis_type == "Catégorielle":
    if not categorical_cols_eda:
        st.info("Aucune colonne catégorielle (telle que définie pour le modèle) trouvée dans les données EDA.")
    else:
        selected_var = st.sidebar.selectbox("Choisir variable catégorielle", categorical_cols_eda)
        if selected_var and selected_var in df_eda.columns:
            st.subheader(f"Analyse de : {selected_var}")
            # Bar plot (comme notebook)
            col_counts = df_eda[selected_var].value_counts()
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            bars = col_counts.plot(kind='bar', color='skyblue', ax=ax_bar)
            for bar in bars.patches:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + col_counts.max()*0.01, f'{int(yval):,}', ha='center', va='bottom')
            ax_bar.set_title(f'Distribution de {selected_var}')
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_bar)
            # Pie plot (comme notebook)
            fig_pie, ax_pie = plt.subplots(figsize=(7,5))
            df_eda[selected_var].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax_pie)
            ax_pie.set_title(f'Répartition de {selected_var}')
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)

elif analysis_type == "Numérique":
    if not numeric_cols_eda:
        st.info("Aucune colonne numérique (telle que définie pour le modèle) trouvée dans les données EDA.")
    else:
        selected_var = st.sidebar.selectbox("Choisir variable numérique", numeric_cols_eda)
        if selected_var and selected_var in df_eda.columns:
            st.subheader(f"Analyse de : {selected_var}")
            # Box plot (Plotly GO comme notebook)
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df_eda[selected_var], name=selected_var, marker=dict(color='skyblue')))
            fig_box.update_layout(title=f'Box plot de {selected_var}')
            st.plotly_chart(fig_box, use_container_width=True)
            # Histogramme (Seaborn)
            fig_hist, ax_hist = plt.subplots(figsize=(10,6))
            sns.histplot(df_eda[selected_var], kde=True, ax=ax_hist, color='lightgreen')
            ax_hist.set_title(f'Distribution de {selected_var}')
            st.pyplot(fig_hist)

st.markdown("<h3 style='text-align : center;'>Tableau de Fréquence</h3>",unsafe_allow_html=True)
st.dataframe(df_eda[selected_var].value_counts().reset_index().rename(columns={'index': selected_var, selected_var: 'Fréquence'}))