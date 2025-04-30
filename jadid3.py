OPTION 1 — Spline lissée (UnivariateSpline)
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import re
import warnings

warnings.filterwarnings("ignore")

# === 1. Charger ton dataset ===
data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],  # Anomalie potentielle
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401]
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
        continue

    try:
        x = group['pillar_num'].values
        y = group['shock'].values
        spline = UnivariateSpline(x, y, s=0.0001)
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

df_result = pd.concat(results, ignore_index=True)
df_result.to_csv("anomalies_detectees_spline_lissee.csv", index=False)
print(df_result[df_result['is_anomaly']])


OPTION 2 — Régression polynomiale (Polyfit, degré 3)
import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

# === 1. Charger ton dataset ===
data = [
    ['10Y', -0.007000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['15Y', -0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['18M', 0.015000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],
    ['1Y', 0.030000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401],  # Anomalie potentielle
    ['2Y', 0.014000, 'OT#DRM EMTN ID IBRD', 20240719, 20250401]
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
        continue

    try:
        x = group['pillar_num'].values
        y = group['shock'].values
        coeffs = np.polyfit(x, y, deg=3)
        poly = np.poly1d(coeffs)
        fitted = poly(x)
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

df_result = pd.concat(results, ignore_index=True)
df_result.to_csv("anomalies_detectees_polyfit.csv", index=False)
print(df_result[df_result['is_anomaly']])
