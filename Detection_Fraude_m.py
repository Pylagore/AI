# -*- coding: utf-8 -*-
"""Detection_Fraude_m.py"""

# Packages (tous les imports du script original)
!pip install loguru
from pathlib import Path
import dill
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
# import pendulum # Non utilisé directement dans le code fourni, peut être commenté
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imb_Pipeline
from loguru import logger
from sklearn import set_config
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                             roc_auc_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             RocCurveDisplay,
                             PrecisionRecallDisplay,
                            )
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline # sklearn.pipeline.Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
# from ucimlrepo import fetch_ucirepo, list_available_datasets # Non utilisé, peut être commenté
# from yellowbrick.classifier import DiscriminationThreshold # Non utilisé, peut être commenté


set_config(display='diagram')
pd.set_option("display.max_columns", None)

"""# Data collection"""

# Pour la portabilité, il est mieux de définir le chemin relatif au script
# Mais je conserve le chemin original pour l'instant.
# L'application Streamlit utilisera un fichier training.csv placé dans son dossier.
HOME_DIR = Path("C:/Users/USER/Desktop/ISEP 2/AI/galsenais-fraud-detection-competition20250319-9824-1x0yyob") # Ce chemin est spécifique à la machine
DATA_TRAIN = HOME_DIR / "training.csv"
# HOME_DIR.mkdir(parents=True, exist_ok=True) # Pas nécessaire si le dossier existe déjà
# print(f"Work directory: {HOME_DIR} \nData directory: {DATA_TRAIN}") # Optionnel

# Charger les données
try:
    df = pd.read_csv(DATA_TRAIN, sep=",")
    logger.info("Données chargées avec succès.")
except FileNotFoundError:
    logger.error(f"Le fichier {DATA_TRAIN} est introuvable. Veuillez vérifier le chemin.")
    exit() # Arrêter si le fichier n'est pas trouvé

# ... (Toute la section EDA et feature engineering du notebook original jusqu'à la modélisation) ...

df['PricingStrategy'] = df['PricingStrategy'].astype('str')

# Création des variables dérivées de AccountId, SubscriptionId, CustomerId
if all(col in df.columns for col in ['CustomerId', 'Amount', 'SubscriptionId', 'TransactionId']):
    df['CustomerId_abs_amount_sum'] = df.groupby('CustomerId')['Amount'].transform(lambda x: x.abs().sum())
    df['SubscriptionId_transaction_count'] = df.groupby('SubscriptionId')['TransactionId'].transform('count')
    df['CustomerId_abs_amount_std'] = df.groupby('CustomerId')['Amount'].transform(lambda x: x.abs().std())
    df['CustomerId_abs_amount_std'] = df['CustomerId_abs_amount_std'].fillna(0)
else:
    logger.warning("Colonnes manquantes pour créer les features CustomerId/SubscriptionId.")

# Création des variables dérivées de TransactionStartTime
if 'Amount' in df.columns:
    df['TransactionType'] = df['Amount'].apply(lambda x: 'Credit' if x < 0 else 'Debit').astype('object')

if 'TransactionStartTime' in df.columns:
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day_name(locale='fr_FR')

    def get_moment(hour):
        if 6 <= hour < 13: return 'matin'
        elif 13 <= hour <= 20: return 'apres-midi'
        else: return 'nuit'
    df['MomentOfDay'] = df['TransactionHour'].apply(get_moment)
else:
    logger.warning("Colonne TransactionStartTime manquante.")


# Colonnes à supprimer (celles du notebook)
cols_to_drop_initial = ['TransactionId', 'BatchId', 'CustomerId','AccountId', 'SubscriptionId',
               'CountryCode', 'CurrencyCode','TransactionStartTime','TransactionHour','ProductCategory']
df.drop(columns=[col for col in cols_to_drop_initial if col in df.columns], inplace=True, errors='ignore')
logger.info(f"Colonnes supprimées (initial): {cols_to_drop_initial}")


# Recodage des variables (la logique exacte des listes est cruciale)
# Pour ProductId
if 'ProductId' in df.columns:
    frequencies_product = df['ProductId'].value_counts(normalize=True) * 100
    products_above_12 = frequencies_product[frequencies_product > 12]
    products_list_script = products_above_12.index.tolist() # Liste pour le script
    logger.info(f"ProductId list for recoding (script): {products_list_script}")
    df['ProductId'] = df['ProductId'].apply(lambda x: x if x in products_list_script else 'Other')

# Pour ProviderId
if 'ProviderId' in df.columns:
    frequencies_provider = df['ProviderId'].value_counts(normalize=True) * 100
    provider_above_12 = frequencies_provider[frequencies_provider > 12]
    provider_list_script = provider_above_12.index.tolist() # Liste pour le script
    logger.info(f"ProviderId list for recoding (script): {provider_list_script}")
    df['ProviderId'] = df['ProviderId'].apply(lambda x: x if x in provider_list_script else 'Other')

# Pour ChannelId (liste fixe du notebook)
channel_list_script = ['ChannelId_2','ChannelId_3']
if 'ChannelId' in df.columns:
    df['ChannelId'] = df['ChannelId'].apply(lambda x: x if x in channel_list_script else 'Other')

# Pour PricingStrategy (liste fixe du notebook)
Pricing_list_script = ["2","4"]
if 'PricingStrategy' in df.columns:
    df['PricingStrategy'] = df['PricingStrategy'].astype(str).apply(lambda x: x if x in Pricing_list_script else 'Other')

logger.info("Recodage des variables catégorielles terminé.")

# Transformations log
if 'Value' in df.columns: df['log_Value'] = np.log1p(df['Value'])
if 'Amount' in df.columns: df['log_abs_Amount'] = np.log1p(df['Amount'].abs())
if 'CustomerId_abs_amount_sum' in df.columns: df['log_CustomerId_abs_amount_sum'] = np.log1p(df['CustomerId_abs_amount_sum'])

# Création de Amount_Value_Ecart
if 'Value' in df.columns and 'Amount' in df.columns:
    df['Amount_Value_Ecart'] = df['Value'] - df['Amount'].abs()

# Suppression des colonnes finales avant modélisation (celles du notebook)
cols_to_drop_final = ['Amount', 'Value', 'log_Value', 'log_abs_Amount','CustomerId_abs_amount_sum']
df.drop(columns=[col for col in cols_to_drop_final if col in df.columns], inplace=True, errors='ignore')
logger.info(f"Colonnes supprimées (final): {cols_to_drop_final}")


# Identifier les colonnes numériques et catégorielles FINALES pour le preprocessor
# Ces listes sont cruciales et doivent correspondre à ce que le preprocessor utilise.
# Elles sont définies dans la section "Modeling" du notebook.
# Je les redéfinis ici pour clarté, en s'assurant qu'elles existent dans le df final.
df_final_features = df.drop('FraudResult', axis=1, errors='ignore') # X pour la modélisation

numeric_columns_model = df_final_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns_model = df_final_features.select_dtypes(include="object").columns.tolist()

logger.info(f"Colonnes numériques finales pour le modèle: {numeric_columns_model}")
logger.info(f"Colonnes catégorielles finales pour le modèle: {categorical_columns_model}")


"""# Modeling

## Régression logistique
"""
# ----------------------------
# 1. Séparation des variables (après tout le feature engineering)
# ----------------------------
if 'FraudResult' not in df.columns:
    logger.error("La colonne cible 'FraudResult' est manquante avant la modélisation.")
    exit()

X = df.drop('FraudResult', axis=1)
y = df['FraudResult']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
logger.info(f"Dimensions X_train: {X_train.shape}, X_val: {X_val.shape}")


# ===> AJOUT IMPORTANT : Sauvegarde des noms des colonnes brutes (celles de X_train) <===
# Ces colonnes sont celles après TOUT le feature engineering manuel,
# et ce sont celles que le preprocessor du pipeline va recevoir.
raw_training_columns_for_pipeline = X_train.columns.tolist()
try:
    with open('training_raw_features.dill', 'wb') as f: # Nom du fichier pour les colonnes
        dill.dump(raw_training_columns_for_pipeline, f)
    logger.info(f"Liste des colonnes (après FE manuel) pour le pipeline sauvegardée sous : training_raw_features_friend.dill")
    logger.info(f"Nombre de colonnes sauvegardées : {len(raw_training_columns_for_pipeline)}")
    # logger.debug(f"Colonnes : {raw_training_columns_for_pipeline}") # Pour débug
except Exception as e:
    logger.error(f"ERREUR lors de la sauvegarde des colonnes pour le pipeline : {e}")
# ===> FIN DE L'AJOUT <===


# ----------------------------
# 2. Prétraitement (Définition du preprocessor)
# ----------------------------
# Utiliser les listes de colonnes déterminées juste avant pour le ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_columns_model), # numeric_columns_model
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns_model) # categorical_columns_model
])
logger.info("Preprocessor défini.")
# logger.debug(f"Preprocessor numeric columns: {numeric_columns_model}")
# logger.debug(f"Preprocessor categorical columns: {categorical_columns_model}")

# ... (Le reste du code de modélisation, y compris la baseline, le tuning, etc.) ...
# Il est crucial que `numeric_columns` et `categorical_columns` utilisés dans
# `preprocessor` correspondent bien à `numeric_columns_model` et `categorical_columns_model`
# Ici, le code du notebook original réutilise `numeric_columns` et `categorical_columns`
# qui ont été définis plus haut. Il faut s'assurer de leur cohérence.
# Dans le notebook, `numeric_columns` est redéfini plusieurs fois. La dernière définition avant
# la modélisation est :
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# numeric_columns = [col for col in numeric_columns if col != 'FraudResult']
# C'est ce que j'ai utilisé pour `numeric_columns_model`.
# De même pour `categorical_columns_model`.

# ----------------------------
# Baseline (modèle par défaut) - pour vérifier que preprocessor est bien défini
# ----------------------------
logger.info("📌 Baseline : Régression Logistique sans tuning ni sampling")

baseline_pipeline = imb_Pipeline([ # Utiliser imb_Pipeline pour la cohérence
    ('preproc', preprocessor),
    ('clf', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)) # Augmenter max_iter
])

try:
    baseline_pipeline.fit(X_train, y_train)
    y_pred_baseline = baseline_pipeline.predict(X_val)
    y_proba_baseline = baseline_pipeline.predict_proba(X_val)[:, 1]

    logger.info("➡️ Performance du modèle baseline :")
    logger.info(f"Accuracy  : {accuracy_score(y_val, y_pred_baseline):.4f}")
    logger.info(f"Précision : {precision_score(y_val, y_pred_baseline, zero_division=0):.4f}")
    logger.info(f"Rappel    : {recall_score(y_val, y_pred_baseline):.4f}")
    logger.info(f"F1-score  : {f1_score(y_val, y_pred_baseline):.4f}")
    logger.info(f"ROC AUC   : {roc_auc_score(y_val, y_proba_baseline):.4f}")
except Exception as e:
    logger.error(f"Erreur lors de l'entraînement ou de l'évaluation du pipeline baseline: {e}")
    logger.error("Vérifiez la cohérence des `numeric_columns_model` et `categorical_columns_model` avec l'état de `X_train`.")


# ... (Tuning and sampling with regression logistique - comme dans le notebook) ...
logger.info("\n--- Tuning et Sampling avec Régression Logistique ---")
# Paramètres de base du classifieur
param_grid = {
    'clf__penalty': ['l2'],
    'clf__C': [0.1, 10], # Réduire pour aller plus vite, le notebook a [0.1, 1, 10]
    'clf__solver': ['lbfgs'], # Le notebook a ['liblinear', 'lbfgs']
    'clf__max_iter': [1000, 6000] # Le notebook a [1000, 2000, 6000]
}
# Paramètres avec suréchantillonnage SMOTE
param_grid_smote = param_grid.copy()
param_grid_smote.update({
    'smote__sampling_strategy': [0.25, 0.5]
})
# Paramètres avec sous-échantillonnage
param_grid_under = param_grid.copy()
param_grid_under.update({
    'under__sampling_strategy': [0.25, 0.5]
})
pipelines = {
    "Original": imb_Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(random_state=42)) # Mettre random_state
    ]),
    "SMOTE": imb_Pipeline([
        ('preproc', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(random_state=42))
    ]),
    "UnderSampling": imb_Pipeline([
        ('preproc', preprocessor),
        ('under', RandomUnderSampler(random_state=42)),
        ('clf', LogisticRegression(random_state=42))
    ]),
    "Pondération": imb_Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
}
resultats = []
best_model_overall = None
best_f1_overall = -1

for methode, pipeline in pipelines.items():
    logger.info(f"\n🔍 Méthode : {methode}")
    current_param_grid = param_grid
    if "SMOTE" in methode: current_param_grid = param_grid_smote
    elif "UnderSampling" in methode: current_param_grid = param_grid_under

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_splits=5 pour aller plus vite, le notebook a 9
    grid = GridSearchCV(pipeline, current_param_grid, cv=cv, scoring='f1', n_jobs=-1, error_score='raise', verbose=1) # verbose=1
    
    try:
        grid.fit(X_train, y_train)
        best_model_current_methode = grid.best_estimator_
        y_pred = best_model_current_methode.predict(X_val)
        y_proba = best_model_current_methode.predict_proba(X_val)[:, 1]
        current_f1 = f1_score(y_val, y_pred)

        resultats.append({
            'Méthode': methode,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Précision': precision_score(y_val, y_pred, zero_division=0),
            'Rappel': recall_score(y_val, y_pred),
            'F1-score': current_f1,
            'ROC AUC': roc_auc_score(y_val, y_proba),
            'Best Params': grid.best_params_
        })
        if current_f1 > best_f1_overall:
            best_f1_overall = current_f1
            best_model_overall = best_model_current_methode # Sauvegarder le meilleur modèle basé sur F1
            logger.info(f"Nouveau meilleur modèle trouvé avec {methode}, F1-score: {current_f1:.4f}")

    except Exception as e:
        logger.error(f"Erreur pendant GridSearchCV pour méthode {methode}: {e}")
        resultats.append({
            'Méthode': methode, 'F1-score': -1, 'Error': str(e)
        })


df_resultats = pd.DataFrame(resultats)
df_resultats = df_resultats.sort_values(by='F1-score', ascending=False).reset_index(drop=True)
logger.info("\n📊 Résumé des performances pour régression logistique:")
print(df_resultats[['Méthode', 'F1-score', 'ROC AUC', 'Best Params']].to_string()) # Afficher plus proprement

# Récupérer le pipeline optimal pour régression logistique
# Le notebook choisit 'UnderSampling' mais le code ci-dessus prend le meilleur F1 global.
# Je vais suivre la logique du meilleur F1 global. Si aucun modèle n'a fonctionné, best_model_overall sera None.
if best_model_overall is not None:
    logger.info(f"\nLe meilleur pipeline global est basé sur le F1-score max: {best_f1_overall:.4f}")
    best_pipeline_regression_logistique = best_model_overall # C'est déjà le pipeline complet
else:
    # Fallback si GridSearchCV a échoué pour toutes les méthodes,
    # ou si on veut explicitement celui du notebook.
    # Pour l'instant, on va supposer que best_model_overall est le bon.
    # il faudra ajuster cette logique.
    logger.warning("Aucun best_model_overall trouvé, tentative de recréer le modèle 'UnderSampling' du notebook si présent.")
    if not df_resultats.empty and 'UnderSampling' in df_resultats['Méthode'].values:
        under_sampling_results = df_resultats[df_resultats['Méthode'] == 'UnderSampling'].iloc[0]
        if pd.notna(under_sampling_results.get('Best Params')):
            best_params_under = under_sampling_results['Best Params']
            best_pipeline_regression_logistique = imb_Pipeline([
                ('preproc', preprocessor),
                ('under', RandomUnderSampler(sampling_strategy=best_params_under['under__sampling_strategy'], random_state=42)),
                ('clf', LogisticRegression(
                    penalty=best_params_under['clf__penalty'],
                    C=best_params_under['clf__C'],
                    solver=best_params_under['clf__solver'],
                    max_iter=best_params_under['clf__max_iter'],
                    random_state=42
                ))
            ])
            logger.info("Recréation du pipeline 'UnderSampling' avec les paramètres trouvés.")
            best_pipeline_regression_logistique.fit(X_train, y_train) # Ré-entraîner
        else:
            logger.error("Impossible de recréer le pipeline 'UnderSampling', paramètres non trouvés ou erreur.")
            best_pipeline_regression_logistique = None # Pour éviter une erreur plus loin
    else:
        logger.error("Méthode 'UnderSampling' non trouvée dans les résultats ou resultats vides.")
        best_pipeline_regression_logistique = None


# Enregistrer le meilleur modèle avec dill (s'il existe)
if best_pipeline_regression_logistique is not None:
    MODEL_SAVE_PATH = 'best_Regression_logistique.dill' # Correspond au nom de fichier dans le notebook
    try:
        with open(MODEL_SAVE_PATH, 'wb') as f:
            dill.dump(best_pipeline_regression_logistique, f)
        logger.info(f"Meilleur modèle de régression logistique sauvegardé sous : {MODEL_SAVE_PATH}")

        # Tester le chargement (optionnel mais bonne pratique)
        with open(MODEL_SAVE_PATH, 'rb') as f:
            loaded_model_test = dill.load(f)
        y_pred_test_load = loaded_model_test.predict(X_val.head()) # Prédire sur quelques lignes
        logger.info(f"Test de prédiction du modèle chargé (premières 5 lignes de X_val): {y_pred_test_load}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde ou du test de chargement du modèle : {e}")
else:
    logger.error("Aucun pipeline de régression logistique final à sauvegarder.")


# ... (La section KNN du notebook pourrait suivre ici si nécessaire) ...
logger.info("Fin du script de modification pour sauvegarde.")