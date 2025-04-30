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
