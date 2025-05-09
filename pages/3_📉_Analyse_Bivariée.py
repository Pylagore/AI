import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway
import numpy as np

st.set_page_config(page_title="Analyse Bivariée ", page_icon="📉", layout="wide")
st.markdown("<h1 style='text-align : center;'>Analyse Bivariée (vs FraudResult)</h1>",unsafe_allow_html=True)

if 'df_processed_for_eda' not in st.session_state or st.session_state['df_processed_for_eda'] is None:
    st.error("Données EDA non disponibles.")
    st.stop()
df_eda = st.session_state['df_processed_for_eda']
if 'FraudResult' not in df_eda.columns: st.error("Cible 'FraudResult' manquante."); st.stop()

raw_cols_for_pipeline = st.session_state.get('raw_training_cols_for_pipeline', df_eda.columns.drop('FraudResult', errors='ignore').tolist())
temp_df_for_type_detection = df_eda[raw_cols_for_pipeline]
numeric_cols_eda = temp_df_for_type_detection.select_dtypes(include=np.number).columns.tolist()
categorical_cols_eda = temp_df_for_type_detection.select_dtypes(include='object').columns.tolist()

analysis_choice = st.sidebar.radio("Analyse", ["Catégorielle vs Cible", "Numérique vs Cible", "Corrélations Numériques"])

if analysis_choice == "Catégorielle vs Cible":
    if not categorical_cols_eda: st.info("Aucune colonne catégorielle pertinente trouvée.")
    else:
        selected_cat = st.sidebar.selectbox("Var. Catégorielle", categorical_cols_eda)
        if selected_cat and selected_cat in df_eda.columns:
            st.subheader(f"'{selected_cat}' vs 'FraudResult'")
            # Barres stackées Plotly
            counts_df = df_eda.groupby([selected_cat, 'FraudResult']).size().reset_index(name='Count')
            fig_bar = px.bar(counts_df, x=selected_cat, y='Count', color='FraudResult', barmode='stack', title=f'{selected_cat} vs FraudResult')
            st.plotly_chart(fig_bar, use_container_width=True)
            # Test Chi2
            try:
                ct = pd.crosstab(df_eda['FraudResult'], df_eda[selected_cat])
                chi2, p, dof, exp = chi2_contingency(ct)
                n_total = ct.sum().sum()
                cramer_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
                st.write(f"Chi2: {chi2:.2f}")
                st.write(f"P-val: {p:.3g}")
                st.write(f"Cramer's V: {cramer_v:.3f}")
            except Exception as e: st.error(f"Erreur Chi2: {e}")

elif analysis_choice == "Numérique vs Cible":
    if not numeric_cols_eda: st.info("Aucune colonne numérique pertinente trouvée.")
    else:
        selected_num = st.sidebar.selectbox("Var. Numérique", numeric_cols_eda)
        if selected_num and selected_num in df_eda.columns:
            st.subheader(f"'{selected_num}' vs 'FraudResult'")
            # Box plot Plotly
            fig_box = px.box(df_eda, x='FraudResult', y=selected_num, title=f'{selected_num} vs FraudResult')
            st.plotly_chart(fig_box, use_container_width=True)
            # Histplot Seaborn
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(data=df_eda, x=selected_num, hue="FraudResult", kde=True, stat="density", common_norm=False, ax=ax_hist)
            ax_hist.set_title(f'Distribution de {selected_num} par FraudResult')
            st.pyplot(fig_hist)
            # Test ANOVA
            try:
                g0 = df_eda[df_eda['FraudResult'] == 0][selected_num].dropna()
                g1 = df_eda[df_eda['FraudResult'] == 1][selected_num].dropna()
                if len(g0) > 1 and len(g1) > 1:
                    f_stat, p_anova = f_oneway(g0, g1)
                    st.write(f"ANOVA - F-stat: {f_stat:.2f}, P-val: {p_anova:.3g}")
                else: st.info("Pas assez de données pour ANOVA.")
            except Exception as e: st.error(f"Erreur ANOVA: {e}")

elif analysis_choice == "Corrélations Numériques":
    st.subheader("Heatmap Corrélations (Pearson)")
    if numeric_cols_eda:
        corr_df = df_eda[numeric_cols_eda].corr(method="pearson")
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_df, cmap='Greens', annot=True, fmt='.2f', mask=np.triu(np.ones_like(corr_df, dtype=bool)), ax=ax_heatmap)
        st.pyplot(fig_heatmap)
    else: st.info("Aucune colonne numérique pour la corrélation.")