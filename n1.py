import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 1. Charger vos deux jeux de données
# ----------------------------------------------------------------------------
# df_filtre : dataset initial
# dfn1      : dataset n+1
# Assurez-vous que les deux DataFrames existent déjà dans votre environnement.

# ----------------------------------------------------------------------------
# 2. Fonction de conversion de 'pillars' en mois
# ----------------------------------------------------------------------------
def convert_pillar_to_months(pillar):
    p = str(pillar).strip().upper()
    m = re.findall(r"\d+\.?\d*", p)
    if not m:
        return np.nan
    value = float(m[0])
    if 'Y' in p:
        return value * 12
    elif 'M' in p and 'Y' not in p:
        return value
    elif 'W' in p:
        return value / 4.33   # 1 mois ≈ 4.33 semaines
    elif 'D' in p:
        return value / 30.0   # 1 mois ≈ 30 jours
    else:
        return np.nan

for df in (df_filtre, dfn1):
    df['pillar_months'] = df['pillars'].apply(convert_pillar_to_months)

# ----------------------------------------------------------------------------
# 3. Préparation des features pour le clustering
# ----------------------------------------------------------------------------
features = ['shock', 'pillar_months', 'scenario_id']  # scenario_id est déjà numérique

# On enlève les lignes incomplètes
df_train = df_filtre.dropna(subset=features).copy()
X_train = df_train[features].values

# ----------------------------------------------------------------------------
# 4. Standardisation
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ----------------------------------------------------------------------------
# 5. Entraînement du modèle K-Means
# ----------------------------------------------------------------------------
k_opt = 3  # à ajuster selon vos analyses précédentes
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df_train['cluster'] = kmeans.fit_predict(X_train_scaled)

# ----------------------------------------------------------------------------
# 6. Attribution du cluster majoritaire par rf_struct_id sur le jeu initial
# ----------------------------------------------------------------------------
rf_old = (
    df_train
    .groupby('rf_struct_id')['cluster']
    .agg(lambda x: x.mode()[0])
    .reset_index()
    .rename(columns={'cluster': 'cluster_old'})
)

# ----------------------------------------------------------------------------
# 7. Application du même modèle au dataset n+1
# ----------------------------------------------------------------------------
dfn1_feat = dfn1.dropna(subset=features).copy()
X_n1_scaled = scaler.transform(dfn1_feat[features].values)
dfn1_feat['cluster'] = kmeans.predict(X_n1_scaled)

# Cluster majoritaire par rf_struct_id pour n+1
rf_new = (
    dfn1_feat
    .groupby('rf_struct_id')['cluster']
    .agg(lambda x: x.mode()[0])
    .reset_index()
    .rename(columns={'cluster': 'cluster_new'})
)

# ----------------------------------------------------------------------------
# 8. Comparaison et détection d'anomalies
# ----------------------------------------------------------------------------
df_compare = rf_old.merge(rf_new, on='rf_struct_id')
df_compare['anomaly'] = df_compare['cluster_old'] != df_compare['cluster_new']

# Afficher le résultat
print(df_compare)

# Exemple de filtre pour ne voir que les anomalies
anomalies = df_compare[df_compare['anomaly']]
print("\nAnomalies (changement de cluster) :")
print(anomalies)
