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
