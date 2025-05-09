import pandas as pd
import numpy as np
from loguru import logger


channel_list_recodage  = ['ChannelId_2','ChannelId_3'] # Fixe du notebook
pricing_list_recodage  = ["2","4"] # Fixe du notebook

def apply_feature_engineering_streamlit(df_input, products_list_arg, provider_list_arg):
    df = df_input.copy()
    logger.info(f"Shape initiale pour feature engineering: {df.shape}")
    logger.debug(f"Colonnes initiales: {df.columns.tolist()}")


    if 'PricingStrategy' in df.columns:
        df['PricingStrategy'] = df['PricingStrategy'].astype('str')

    # Création des variables dérivées (nécessite les colonnes originales)
    original_cols_needed = ['CustomerId', 'Amount', 'SubscriptionId', 'TransactionId', 'TransactionStartTime']
    missing_original_for_fe = [col for col in original_cols_needed if col not in df.columns]
    if missing_original_for_fe:
        logger.warning(f"Colonnes originales manquantes pour feature engineering: {missing_original_for_fe}. Certaines features ne seront pas créées.")

    if 'CustomerId' in df.columns and 'Amount' in df.columns:
        df['CustomerId_abs_amount_sum'] = df.groupby('CustomerId')['Amount'].transform(lambda x: x.abs().sum())
        df['CustomerId_abs_amount_std'] = df.groupby('CustomerId')['Amount'].transform(lambda x: x.abs().std()).fillna(0)
    else:
        df['CustomerId_abs_amount_sum'] = 0 # Valeur par défaut
        df['CustomerId_abs_amount_std'] = 0 # Valeur par défaut

    if 'SubscriptionId' in df.columns and 'TransactionId' in df.columns:
        df['SubscriptionId_transaction_count'] = df.groupby('SubscriptionId')['TransactionId'].transform('count')
    else:
        df['SubscriptionId_transaction_count'] = 0 # Valeur par défaut

    if 'Amount' in df.columns:
        df['TransactionType'] = df['Amount'].apply(lambda x: 'Credit' if x < 0 else 'Debit').astype('object')
    else:
        df['TransactionType'] = 'Unknown' # Valeur par défaut

    if 'TransactionStartTime' in df.columns:
        try:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day_name(locale='fr_FR')
            def get_moment(hour):
                if 6 <= hour < 13: return 'matin'
                elif 13 <= hour <= 20: return 'apres-midi'
                else: return 'nuit'
            df['MomentOfDay'] = df['TransactionHour'].apply(get_moment)
        except Exception as e:
            logger.error(f"Erreur conversion TransactionStartTime ou création features temporelles: {e}")
            if 'TransactionHour' not in df.columns : df['TransactionHour'] = 0
            if 'TransactionDay' not in df.columns : df['TransactionDay'] = 'Unknown'
            if 'MomentOfDay' not in df.columns : df['MomentOfDay'] = 'Unknown'
    else: # S'assurer que les colonnes existent même si TransactionStartTime est absente
        if 'TransactionHour' not in df.columns : df['TransactionHour'] = 0
        if 'TransactionDay' not in df.columns : df['TransactionDay'] = 'Unknown'
        if 'MomentOfDay' not in df.columns : df['MomentOfDay'] = 'Unknown'


    # Colonnes à supprimer (celles du notebook, attention à l'ordre)
    cols_to_drop_initial = ['TransactionId', 'BatchId', 'CustomerId','AccountId', 'SubscriptionId',
                   'CountryCode', 'CurrencyCode','TransactionStartTime','TransactionHour','ProductCategory']
    df.drop(columns=[col for col in cols_to_drop_initial if col in df.columns], inplace=True, errors='ignore')
    logger.info(f"Colonnes après suppression initiale: {df.columns.tolist()}")

    # Recodage des variables
    if 'ProductId' in df.columns and products_list_arg:
        df['ProductId'] = df['ProductId'].apply(lambda x: x if x in products_list_arg else 'Other')
    if 'ProviderId' in df.columns and provider_list_arg:
        df['ProviderId'] = df['ProviderId'].apply(lambda x: x if x in provider_list_arg else 'Other')
    if 'ChannelId' in df.columns:
        df['ChannelId'] = df['ChannelId'].apply(lambda x: x if x in channel_list_recodage else 'Other')
    if 'PricingStrategy' in df.columns:
        df['PricingStrategy'] = df['PricingStrategy'].astype(str).apply(lambda x: x if x in pricing_list_recodage else 'Other')

    # Transformations log
    for col_base, col_log in [('Value', 'log_Value'),
                              ('Amount', 'log_abs_Amount'), # Note: Amount est droppé plus tard
                              ('CustomerId_abs_amount_sum', 'log_CustomerId_abs_amount_sum')]:
        if col_base in df.columns:
            if col_base == 'Amount':
                df[col_log] = np.log1p(df[col_base].abs())
            else:
                df[col_log] = np.log1p(df[col_base])
        elif col_log not in df.columns: # Créer avec 0 si la base n'existe pas et log non plus
             df[col_log] = 0


    # Création de Amount_Value_Ecart
    if 'Value' in df.columns and 'Amount' in df.columns: # Amount est peut-être déjà transformé ou droppé
        df['Amount_Value_Ecart'] = df['Value'] - df['Amount'].abs()
    elif 'Value' in df.columns and 'log_abs_Amount' in df.columns and 'Amount' not in df.columns:
        # Si Amount a été droppé mais Value et log_abs_Amount existent, on ne peut pas le recréer facilement.
        # Le notebook droppe Amount APRES avoir créé Amount_Value_Ecart.
        # Donc, il faut s'assurer que 'Amount' (original) est là pour cette étape.
        # Pour la prédiction, si 'Amount' n'est pas dans les colonnes brutes d'entrée, cette feature sera problématique.
        # On va la mettre à 0 par défaut si les colonnes sources manquent.
        if 'Amount_Value_Ecart' not in df.columns: df['Amount_Value_Ecart'] = 0
        logger.warning("Impossible de calculer 'Amount_Value_Ecart' car 'Amount' est manquant (peut-être déjà transformé/supprimé). Mis à 0.")
    elif 'Amount_Value_Ecart' not in df.columns:
        df['Amount_Value_Ecart'] = 0


    # Suppression des colonnes finales (celles que le notebook supprime en dernier)
    cols_to_drop_final = ['Amount', 'Value', 'log_Value', 'log_abs_Amount','CustomerId_abs_amount_sum']
    df.drop(columns=[col for col in cols_to_drop_final if col in df.columns], inplace=True, errors='ignore')
    logger.info(f"Colonnes après suppression finale: {df.columns.tolist()}")
    logger.info(f"Shape finale après feature engineering: {df.shape}")

    return df