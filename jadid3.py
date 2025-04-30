import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import re
import warnings

warnings.filterwarnings("ignore")

# Exemple de données (à remplacer par ton propre fichier CSV)
data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['6M', 0.000115, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['6Y', 0.000101, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['7Y', 0.000094, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['8Y', 0.000083, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['9Y', 0.000063, 'OT#RISK2_JPY3M_CME', 20250307, 20250401]
]

df = pd.DataFrame(data, columns=['pillars', 'shock', 'rf_struct_id', 'scenario_id', 'as_of_date'])

# Fonction pour convertir 'pillars' en années décimales
def convert_pillar(pillar):
    if re.match(r'^\d+Y$', pillar):
        return int(pillar[:-1])
    elif re.match(r'^\d+M$', pillar):
        return int(pillar[:-1]) / 12
    return np.nan

df['pillar_num'] = df['pillars'].apply(convert_pillar)

# Exemple sur un seul groupe : rf_struct_id + scenario_id + as_of_date
group_key = ('OT#DRM EMTN ID IBRD', 20240719, 20250401)
df_group = df[(df['rf_struct_id'] == group_key[0]) &
              (df['scenario_id'] == group_key[1]) &
              (df['as_of_date'] == group_key[2])]

# Tri par pillar
df_group = df_group.sort_values('pillar_num')

# Ajustement spline cubique
x = df_group['pillar_num'].values
y = df_group['shock'].values
spline = CubicSpline(x, y)

# Valeurs ajustées
df_group['fitted_shock'] = spline(x)
df_group['residual'] = df_group['shock'] - df_group['fitted_shock']
df_group['z_score'] = (df_group['residual'] - df_group['residual'].mean()) / df_group['residual'].std()

# Seuil d’anomalie : |z| > 2
df_group['is_anomaly'] = df_group['z_score'].abs() > 2

# Affichage du résultat
print(df_group)

*************************************

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import re
import warnings

warnings.filterwarnings("ignore")

# === 1. Charger ton dataset ===
# Remplace ce chemin par ton propre fichier CSV
# df = pd.read_csv("chemin/vers/ton_fichier.csv")
# Pour l'exemple, on part d'un jeu fictif :

data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['6M', 0.000115, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['6Y', 0.000101, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['7Y', 0.000094, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['8Y', 0.000083, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['9Y', 0.000063, 'OT#RISK2_JPY3M_CME', 20250307, 20250401]
]

df = pd.DataFrame(data, columns=['pillars', 'shock', 'rf_struct_id', 'scenario_id', 'as_of_date'])

# === 2. Transformation des pillars en format numérique (en années) ===
def convert_pillar(pillar):
    if re.match(r'^\d+Y$', pillar):
        return int(pillar[:-1])
    elif re.match(r'^\d+M$', pillar):
        return int(pillar[:-1]) / 12
    return np.nan

df['pillar_num'] = df['pillars'].apply(convert_pillar)

# === 3. Appliquer la détection d’anomalie par groupe ===

results = []

grouped = df.groupby(['rf_struct_id', 'scenario_id', 'as_of_date'])

for name, group in grouped:
    group = group.dropna(subset=['pillar_num']).sort_values('pillar_num')
    if len(group) < 4:
        continue  # pas assez de points pour faire une spline fiable

    try:
        x = group['pillar_num'].values
        y = group['shock'].values
        spline = CubicSpline(x, y)
        fitted = spline(x)
        residuals = y - fitted
        z_scores = (residuals - residuals.mean()) / residuals.std()

        group = group.copy()
        group['fitted_shock'] = fitted
        group['residual'] = residuals
        group['z_score'] = z_scores
        group['is_anomaly'] = np.abs(z_scores) > 2

        results.append(group)
    except Exception as e:
        print(f"Erreur pour le groupe {name}: {e}")
        continue

# === 4. Fusionner tous les résultats ===
df_result = pd.concat(results, ignore_index=True)

# === 5. Sauvegarde ou affichage ===
df_result.to_csv("anomalies_detectees.csv", index=False)
print(df_result[df_result['is_anomaly']])


//////////////////////////////////////

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import re
import warnings

warnings.filterwarnings("ignore")

# Charger les données (remplacer par ton propre fichier si besoin)
data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['6M', 0.000115, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['6Y', 0.000101, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['7Y', 0.000094, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['8Y', 0.000083, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['9Y', 0.000063, 'OT#RISK2_JPY3M_CME', 20250307, 20250401]
]

df = pd.DataFrame(data, columns=['pillars', 'shock', 'rf_struct_id', 'scenario_id', 'as_of_date'])

# Conversion des pillars en nombre d'années
def convert_pillar(pillar):
    if re.match(r'^\d+Y$', pillar):
        return int(pillar[:-1])
    elif re.match(r'^\d+M$', pillar):
        return int(pillar[:-1]) / 12
    return np.nan

df['pillar_num'] = df['pillars'].apply(convert_pillar)

# Initialisation d'une liste pour stocker les résultats
results = []

# Boucle sur chaque groupe unique
for (rf, scen, date), group in df.groupby(['rf_struct_id', 'scenario_id', 'as_of_date']):
    group = group.dropna(subset=['pillar_num']).sort_values('pillar_num')
    if len(group) < 4:
        continue  # trop peu de points pour une spline fiable

    x = group['pillar_num'].values
    y = group['shock'].values
    spline = CubicSpline(x, y)
    group['fitted_shock'] = spline(x)
    group['residual'] = group['shock'] - group['fitted_shock']
    group['z_score'] = (group['residual'] - group['residual'].mean()) / group['residual'].std()
    group['is_anomaly'] = group['z_score'].abs() > 2
    results.append(group)

# Fusion des résultats
df_all = pd.concat(results)
print(df_all[df_all['is_anomaly'] == True])

**********/////////////////

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import re
import warnings

warnings.filterwarnings("ignore")

# Exemple de données (remplace par pd.read_csv("ton_fichier.csv") si besoin)
data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['6M', 0.000115, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['6Y', 0.000101, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['7Y', 0.000094, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['8Y', 0.000083, 'OT#RISK2_JPY3M_CME', 20250307, 20250401],
    ['9Y', 0.000063, 'OT#RISK2_JPY3M_CME', 20250307, 20250401]
]

df = pd.DataFrame(data, columns=['pillars', 'shock', 'rf_struct_id', 'scenario_id', 'as_of_date'])

# Conversion des pillars en années décimales
def convert_pillar(pillar):
    if re.match(r'^\d+Y$', pillar):
        return int(pillar[:-1])
    elif re.match(r'^\d+M$', pillar):
        return int(pillar[:-1]) / 12
    return np.nan

df['pillar_num'] = df['pillars'].apply(convert_pillar)

# Liste pour stocker les résultats
resultats = []

# Boucle sur chaque groupe unique
for group_key in df.groupby(['rf_struct_id', 'scenario_id', 'as_of_date']).groups.keys():
    df_group = df[(df['rf_struct_id'] == group_key[0]) &
                  (df['scenario_id'] == group_key[1]) &
                  (df['as_of_date'] == group_key[2])]
    
    df_group = df_group.dropna(subset=['pillar_num']).sort_values('pillar_num')
    
    if len(df_group) < 4:
        continue  # trop peu de points pour une spline fiable

    x = df_group['pillar_num'].values
    y = df_group['shock'].values
    spline = CubicSpline(x, y)

    df_group['fitted_shock'] = spline(x)
    df_group['residual'] = df_group['shock'] - df_group['fitted_shock']
    df_group['z_score'] = (df_group['residual'] - df_group['residual'].mean()) / df_group['residual'].std()
    df_group['is_anomaly'] = df_group['z_score'].abs() > 2

    resultats.append(df_group)

# Fusion de tous les groupes
df_final = pd.concat(resultats)

# Affichage des anomalies uniquement
print(df_final[df_final['is_anomaly'] == True])

