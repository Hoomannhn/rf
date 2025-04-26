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

# ****************************************************************************
# a agrégé le résultat par rf_struct_id (mode), ce qui ne vous rend qu’un seul enregistrement par structure de risque (et vous aviez 750 rf_struct_id uniques).
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 1. Charger vos deux jeux de données
#    - df_filtre : dataset t0 (4 millions de lignes)
#    - dfn1      : dataset t+1 (4 millions de lignes, mêmes clés)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 2. Conversion de 'pillars' en variable numérique (mois)
# ----------------------------------------------------------------------------
def convert_pillar_to_months(pillar):
    p = str(pillar).strip().upper()
    m = re.findall(r"\d+\.?\d*", p)
    if not m:
        return np.nan
    v = float(m[0])
    if 'Y' in p:
        return v * 12
    elif 'M' in p and 'Y' not in p:
        return v
    elif 'W' in p:
        return v / 4.33
    elif 'D' in p:
        return v / 30.0
    else:
        return np.nan

for df in (df_filtre, dfn1):
    df['pillar_months'] = df['pillars'].apply(convert_pillar_to_months)

# ----------------------------------------------------------------------------
# 3. Préparation des features
# ----------------------------------------------------------------------------
features = ['shock', 'pillar_months', 'scenario_id']  # scenario_id déjà numérique
df0 = df_filtre.dropna(subset=features).copy()
df1 = dfn1.dropna(subset=features).copy()

# ----------------------------------------------------------------------------
# 4. Standardisation sur le jeu t0
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X0 = scaler.fit_transform(df0[features].values)

# ----------------------------------------------------------------------------
# 5. Entraînement du K-Means sur t0
# ----------------------------------------------------------------------------
k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=0)
df0['cluster_old'] = kmeans.fit_predict(X0)

# ----------------------------------------------------------------------------
# 6. Application du même scaler + modèle sur t+1
# ----------------------------------------------------------------------------
X1 = scaler.transform(df1[features].values)
df1['cluster_new'] = kmeans.predict(X1)

# ----------------------------------------------------------------------------
# 7. Merge ligne-à-ligne pour comparer
#    On part du principe que (rf_struct_id, pillars) sont identiques
#    entre df0 et df1 pour chaque ligne.
# ----------------------------------------------------------------------------
df_cmp = pd.merge(
    df0[['rf_struct_id','pillars','cluster_old']],
    df1[['rf_struct_id','pillars','cluster_new']],
    how='inner',
    on=['rf_struct_id','pillars']
)

# ----------------------------------------------------------------------------
# 8. Marquer les anomalies
# ----------------------------------------------------------------------------
df_cmp['anomaly'] = df_cmp['cluster_old'] != df_cmp['cluster_new']

# ----------------------------------------------------------------------------
# 9. Résultats
# ----------------------------------------------------------------------------
print("Total lignes comparées :", len(df_cmp))
print("Total anomalies détectées :", df_cmp['anomaly'].sum())

# Si vous voulez voir un échantillon :
print(df_cmp[df_cmp['anomaly']].head())

# Et pour avoir la liste complète des anomalies :
anomalies = df_cmp[df_cmp['anomaly']].copy()

