import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Vos deux jeux de données déjà chargés :
#    - df_filtre : dataset t0, colonnes ['rf_struct_id','pillars','shock','as_of_date',…]
#    - dfn1      : dataset t+1, mêmes colonnes
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Concaténer t0 et t+1 en un seul DataFrame
# ----------------------------------------------------------------------------
df_all = pd.concat([df_filtre, dfn1], ignore_index=True)

# ----------------------------------------------------------------------------
# 3. Créer une clé unique pour chaque (rf_struct_id, as_of_date)
# ----------------------------------------------------------------------------
df_all['key'] = df_all['rf_struct_id'].astype(str) + '_' + df_all['as_of_date'].astype(str)

# ----------------------------------------------------------------------------
# 4. Agréger les shocks par key pour obtenir max, min et médiane
# ----------------------------------------------------------------------------
df_feats = (
    df_all
    .groupby('key', as_index=False)
    .agg(
        max_shock    = ('shock', 'max'),
        min_shock    = ('shock', 'min'),
        median_shock = ('shock', 'median'),
        rf_struct_id = ('rf_struct_id', 'first'),
        as_of_date   = ('as_of_date',   'first')
    )
)

# ----------------------------------------------------------------------------
# 5. Préparer et standardiser les features
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
X = df_feats[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 6. Clustering K-Means sur l’ensemble (t0 + t+1)
# ----------------------------------------------------------------------------
k_opt = 3  # ajuster avec méthode du coude / silhouette si besoin
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_feats['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------------------
# 7. Pivot pour regrouper par rf_struct_id et obtenir cluster_old / cluster_new
# ----------------------------------------------------------------------------
df_pivot = df_feats.pivot(index='rf_struct_id', columns='as_of_date', values='cluster').reset_index()

# Identifier simplement les deux dates (dans l’ordre trié)
dates = sorted([c for c in df_pivot.columns if c != 'rf_struct_id'], key=lambda d: pd.to_datetime(d))
date_old, date_new = dates[0], dates[1]

# ----------------------------------------------------------------------------
# 8. Renommer les colonnes pour cluster_old et cluster_new
# ----------------------------------------------------------------------------
df_pivot = df_pivot.rename(columns={
    date_old: 'cluster_old',
    date_new: 'cluster_new'
})

# ----------------------------------------------------------------------------
# 9. Détection d’anomalies : changement de cluster
# ----------------------------------------------------------------------------
df_pivot['anomaly'] = df_pivot['cluster_old'] != df_pivot['cluster_new']

# ----------------------------------------------------------------------------
# 10. DataFrame final
#     colonnes : rf_struct_id, cluster_old, cluster_new, anomaly
# ----------------------------------------------------------------------------
df_result = df_pivot[['rf_struct_id', 'cluster_old', 'cluster_new', 'anomaly']].copy()

print(df_result.head())
print(f"\nNombre de rf_struct_id anomalies : {df_result['anomaly'].sum()}")


# *******************************

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Charger vos deux DataFrames existants :
#    - df_filtre : dataset t0, colonnes ['rf_struct_id','pillars','shock','as_of_date', …]
#    - dfn1      : dataset t+1, mêmes colonnes
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Créer une clé unique pour chaque ligne (clé = rf_struct_id + as_of_date)
# ----------------------------------------------------------------------------
for df in (df_filtre, dfn1):
    df['key'] = (
        df['rf_struct_id'].astype(str)
        + '_' +
        df['as_of_date'].astype(str)
    )

# ----------------------------------------------------------------------------
# 3. Agréger les shocks par key pour générer les features max, min, median
# ----------------------------------------------------------------------------
def make_feats(df):
    return (
        df
        .groupby('key', as_index=False)['shock']
        .agg(
            max_shock    = ('shock', 'max'),
            min_shock    = ('shock', 'min'),
            median_shock = ('shock', 'median'),
            rf_struct_id = ('rf_struct_id', 'first'),
            as_of_date   = ('as_of_date',   'first')
        )
    )

df0_feats = make_feats(df_filtre)  # features pour t0
df1_feats = make_feats(dfn1)       # features pour t+1

# ----------------------------------------------------------------------------
# 4. Concaténer les deux jeux de features pour le clustering
# ----------------------------------------------------------------------------
df_feats_all = pd.concat([df0_feats, df1_feats], ignore_index=True)

# ----------------------------------------------------------------------------
# 5. Préparer et standardiser la matrice de features
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
X = df_feats_all[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 6. Clustering K-Means sur l’ensemble (t0 + t+1)
# ----------------------------------------------------------------------------
k_opt = 3  # ajuster selon votre méthode du coude / silhouette
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_feats_all['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------------------
# 7. Rattacher les clusters à df0_feats (cluster_old) et df1_feats (cluster_new)
# ----------------------------------------------------------------------------
df0 = df0_feats[['key']].merge(df_feats_all[['key','cluster']], on='key')
df0 = df0.rename(columns={'cluster':'cluster_old'})

df1 = df1_feats[['key']].merge(df_feats_all[['key','cluster']], on='key')
df1 = df1.rename(columns={'cluster':'cluster_new'})

# ----------------------------------------------------------------------------
# 8. Déterminer le cluster majoritaire par rf_struct_id pour t0 et t+1
# ----------------------------------------------------------------------------
rf_old = (
    df0
    .merge(df0_feats[['key','rf_struct_id']], on='key')
    .groupby('rf_struct_id')['cluster_old']
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

rf_new = (
    df1
    .merge(df1_feats[['key','rf_struct_id']], on='key')
    .groupby('rf_struct_id')['cluster_new']
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

# ----------------------------------------------------------------------------
# 9. Comparer cluster_old vs. cluster_new par rf_struct_id et marquer anomalies
# ----------------------------------------------------------------------------
df_compare = pd.merge(rf_old, rf_new, on='rf_struct_id', how='inner')
df_compare['anomaly'] = df_compare['cluster_old'] != df_compare['cluster_new']

# ----------------------------------------------------------------------------
# 10. Résultat final
# ----------------------------------------------------------------------------
# Colonnes : rf_struct_id, cluster_old, cluster_new, anomaly
print(df_compare.head())
print(f"\nNombre de rf_struct_id anomalies : {df_compare['anomaly'].sum()}")





######******************************#######




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Charger vos deux DataFrames existants :
#    - df_filtre : dataset t0, colonnes ['rf_struct_id','pillars','shock','as_of_date', …]
#    - dfn1      : dataset t+1, mêmes colonnes
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Pour chaque dataset, créer la clé et agréger les shocks en features
# ----------------------------------------------------------------------------
def make_features(df):
    # 2a. Créer la clé unique
    df = df.copy()
    df['key'] = df['rf_struct_id'].astype(str) + '_' + df['as_of_date'].astype(str)
    # 2b. Agréger shock par key
    df_feats = (
        df
        .groupby('key', as_index=False)
        .agg(
            rf_struct_id = ('rf_struct_id', 'first'),
            as_of_date   = ('as_of_date',   'first'),
            max_shock    = ('shock',        'max'),
            min_shock    = ('shock',        'min'),
            median_shock = ('shock',        'median')
        )
    )
    return df_feats

df0_feats = make_features(df_filtre)  # features pour t0
df1_feats = make_features(dfn1)       # features pour t+1

# ----------------------------------------------------------------------------
# 3. Concaténer les deux jeux de features pour le clustering
# ----------------------------------------------------------------------------
df_feats_all = pd.concat([df0_feats, df1_feats], ignore_index=True)

# ----------------------------------------------------------------------------
# 4. Préparer et standardiser la matrice de features
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
X = df_feats_all[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 5. Clustering K-Means sur l’ensemble (t0 + t+1)
# ----------------------------------------------------------------------------
k_opt = 3  # à ajuster via méthode du coude / silhouette
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_feats_all['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------------------
# 6. Pivot pour obtenir, par rf_struct_id, cluster_old et cluster_new
# ----------------------------------------------------------------------------
df_pivot = (
    df_feats_all
    .pivot(index='rf_struct_id', columns='as_of_date', values='cluster')
    .reset_index()
)

# Identifier les deux dates (triées chronologiquement)
dates = sorted([c for c in df_pivot.columns if c != 'rf_struct_id'], key=lambda d: pd.to_datetime(d))
date_old, date_new = dates[0], dates[1]

# Renommer les colonnes pour plus de clarté
df_pivot = df_pivot.rename(columns={
    date_old: 'cluster_old',
    date_new: 'cluster_new'
})

# ----------------------------------------------------------------------------
# 7. Détection d’anomalies : changement de cluster par rf_struct_id
# ----------------------------------------------------------------------------
df_pivot['anomaly'] = df_pivot['cluster_old'] != df_pivot['cluster_new']

# ----------------------------------------------------------------------------
# 8. DataFrame résultat
#    colonnes : rf_struct_id, cluster_old, cluster_new, anomaly
# ----------------------------------------------------------------------------
df_result = df_pivot[['rf_struct_id', 'cluster_old', 'cluster_new', 'anomaly']].copy()

print(df_result.head())
print(f"\nNombre de rf_struct_id anomalies : {df_result['anomaly'].sum()}")


############################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# 1. Charger vos deux DataFrames existants :
#    - df_filtre : dataset t0, colonnes ['rf_struct_id', 'pillars', 'shock', 'as_of_date', …]
#    - dfn1      : dataset t+1, mêmes colonnes
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Extraire la date t0 et la date t+1
# ----------------------------------------------------------------------------
dates0 = df_filtre['as_of_date'].unique()
dates1 = dfn1   ['as_of_date'].unique()
if len(dates0) != 1 or len(dates1) != 1:
    raise ValueError(f"Chaque dataset doit contenir exactement une date : "
                     f"trouvé {len(dates0)} dans df_filtre, {len(dates1)} dans dfn1.")
date_old = dates0[0]
date_new = dates1[0]

# ----------------------------------------------------------------------------
# 3. Agréger les shocks par rf_struct_id pour t0 et t+1
# ----------------------------------------------------------------------------
def aggregate_shocks(df, as_of_date):
    df_feats = (
        df
        .groupby('rf_struct_id')['shock']
        .agg(
            max_shock='max',
            min_shock='min',
            median_shock='median'
        )
        .reset_index()
    )
    df_feats['as_of_date'] = as_of_date
    return df_feats

df0_feats = aggregate_shocks(df_filtre, date_old)
df1_feats = aggregate_shocks(dfn1,    date_new)

# ----------------------------------------------------------------------------
# 4. Concaténer les deux jeux de features pour le clustering
# ----------------------------------------------------------------------------
df_feats_all = pd.concat([df0_feats, df1_feats], ignore_index=True)

# ----------------------------------------------------------------------------
# 5. Préparer et standardiser la matrice de features
# ----------------------------------------------------------------------------
features = ['max_shock', 'min_shock', 'median_shock']
X = df_feats_all[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 6. Clustering K-Means sur l’ensemble (t0 + t+1)
# ----------------------------------------------------------------------------
k_opt = 3  # ajustez selon méthode du coude / silhouette
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_feats_all['cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------------------
# 7. Pivot pour obtenir, pour chaque rf_struct_id, cluster_old et cluster_new
# ----------------------------------------------------------------------------
df_pivot = (
    df_feats_all
    .pivot_table(
        index='rf_struct_id',          # lignes = rf_struct_id
        columns='as_of_date',          # colonnes = dates
        values='cluster',              # valeur = label de cluster
        aggfunc='first'                # en cas de doublon, on garde le premier
    )
    .reset_index()
)

# ----------------------------------------------------------------------------
# 8. Renommer les colonnes pour plus de clarté
# ----------------------------------------------------------------------------

dates = sorted(
    [c for c in df_pivot.columns if c != 'rf_struct_id'],
    key=lambda d: pd.to_datetime(d)
)
date_old, date_new = dates[0], dates[1]

df_pivot = df_pivot.rename(columns={
    date_old: 'cluster_old',
    date_new: 'cluster_new'
})

df_pivot['anomaly'] = df_pivot['cluster_old'] != df_pivot['cluster_new']

df_result = df_pivot[['rf_struct_id', 'cluster_old', 'cluster_new', 'anomaly']].copy()

print(df_result.head())
print(f"\nNombre de rf_struct_id anomalies : {df_result['anomaly'].sum()}")
