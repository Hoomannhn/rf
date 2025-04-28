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



# ////////////////////////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Charger vos deux jeux de données existants :
#    - df_filtre : dataset t0, contenant ['rf_struct_id', 'pillars', 'shock', 'as_of_date', ...]
#    - dfn1      : dataset t+1, mêmes colonnes (dont 'rf_struct_id', 'shock', 'as_of_date')
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Concaténer les deux jeux de données en un seul DataFrame
# ----------------------------------------------------------------------------
df_all = pd.concat([df_filtre, dfn1], ignore_index=True)

# ----------------------------------------------------------------------------
# 3. Calculer les features agrégées par (rf_struct_id, as_of_date)
# ----------------------------------------------------------------------------
df_feats = (
    df_all
    .groupby(['rf_struct_id', 'as_of_date'])['shock']
    .agg(
        max_shock='max',
        min_shock='min',
        median_shock='median'
    )
    .reset_index()
)

# ----------------------------------------------------------------------------
# 4. Préparation des features pour le clustering
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
X = df_feats[features].values

# ----------------------------------------------------------------------------
# 5. Standardisation
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 6. Clustering K-Means sur l’ensemble des observations
# ----------------------------------------------------------------------------
k_opt = 3  # à ajuster selon vos analyses du coude / silhouette
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_feats['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------------------
# 7. Séparer à nouveau les deux dates pour chaque rf_struct_id
# ----------------------------------------------------------------------------
# Pivot pour avoir un cluster par date en colonne
df_pivot = (
    df_feats
    .pivot(index='rf_struct_id', columns='as_of_date', values='cluster')
    .reset_index()
)

# Renommer les colonnes en fonction des dates
# Supposons que df_filtre['as_of_date'].unique() donne [date0, date1]
date0, date1 = sorted(df_feats['as_of_date'].unique())
df_pivot = df_pivot.rename(columns={
    date0: 'cluster_old',
    date1: 'cluster_new'
})

# ----------------------------------------------------------------------------
# 8. Détection d’anomalies par rf_struct_id
#    Un rf_struct_id est anomalie si son cluster_old != cluster_new
# ----------------------------------------------------------------------------
df_pivot['anomaly'] = df_pivot['cluster_old'] != df_pivot['cluster_new']

# ----------------------------------------------------------------------------
# 9. Résultat final
# ----------------------------------------------------------------------------
# df_pivot contient pour chaque rf_struct_id :
#   - cluster_old (à la date t0)
#   - cluster_new (à la date t+1)
#   - anomaly (True si changement de cluster)
print(df_pivot.head())

# Pour ne garder que les anomalies :
anomalies = df_pivot[df_pivot['anomaly']]
print(f"\nNombre de rf_struct_id anomalies : {len(anomalies)}")
print(anomalies)
