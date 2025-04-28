import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Charger vos deux jeux de données (existants) :
#    - df_filtre : dataset t0, contenant ['rf_struct_id', 'pillars', 'shock', 'as_of_date', ...]
#    - dfn1      : dataset t+1, mêmes colonnes et mêmes rf_struct_id
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Création d’une clé unique pour chaque observation
# ----------------------------------------------------------------------------
for df in (df_filtre, dfn1):
    df['key'] = (
        df['rf_struct_id'].astype(str) + '_' +
        df['pillars'].astype(str)       + '_' +
        df['shock'].astype(str)         + '_' +
        df['as_of_date'].astype(str)
    )

# ----------------------------------------------------------------------------
# 3. Agrégation des shocks par clé pour générer les features
# ----------------------------------------------------------------------------
def compute_shock_features(df):
    return (
        df
        .groupby('key')['shock']
        .agg(
            max_shock='max',
            min_shock='min',
            median_shock='median'
        )
        .reset_index()
    )

df0_feats = compute_shock_features(df_filtre)  # t0
df1_feats = compute_shock_features(dfn1)       # t+1

# ----------------------------------------------------------------------------
# 4. Préparation des features et standardisation (mêmes pour t0 et t+1)
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
scaler = StandardScaler()
X0 = scaler.fit_transform(df0_feats[features])
X1 = scaler.transform(df1_feats[features])

# ----------------------------------------------------------------------------
# 5. Clustering K-Means sur t0, puis application sur t+1
# ----------------------------------------------------------------------------
k_opt = 3  # ajustez en amont si besoin
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df0_feats['cluster_old'] = kmeans.fit_predict(X0)
df1_feats['cluster_new'] = kmeans.predict(X1)

# ----------------------------------------------------------------------------
# 6. Récupération de rf_struct_id depuis la clé
# ----------------------------------------------------------------------------
df0_feats['rf_struct_id'] = df0_feats['key'].str.split('_').str[0]
df1_feats['rf_struct_id'] = df1_feats['key'].str.split('_').str[0]

# ----------------------------------------------------------------------------
# 7. Détermination du cluster majoritaire par rf_struct_id
# ----------------------------------------------------------------------------
df0_rf = (
    df0_feats
    .groupby('rf_struct_id')['cluster_old']
    .agg(cluster_old=lambda x: x.mode()[0])
    .reset_index()
)
df1_rf = (
    df1_feats
    .groupby('rf_struct_id')['cluster_new']
    .agg(cluster_new=lambda x: x.mode()[0])
    .reset_index()
)

# ----------------------------------------------------------------------------
# 8. Comparaison des clusters par rf_struct_id et détection d’anomalies
# ----------------------------------------------------------------------------
df_compare = pd.merge(df0_rf, df1_rf, on='rf_struct_id', how='inner')
df_compare['anomaly'] = df_compare['cluster_old'] != df_compare['cluster_new']

# ----------------------------------------------------------------------------
# 9. Résultat : chaque rf_struct_id, son cluster t0, son cluster t+1, et flag anomalie
# ----------------------------------------------------------------------------
print(df_compare.head())  # aperçu
# Pour ne garder que les anomalies :
anomalies = df_compare[df_compare['anomaly']]
print(f"\nNombre de rf_struct_id anomalies : {len(anomalies)}")
print(anomalies)
