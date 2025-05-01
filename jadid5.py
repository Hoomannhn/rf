import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import re
import warnings

warnings.filterwarnings("ignore")

# === Exemple de dataset ===
data = [
    ['1Y',  0.010, 'RF1', 20240101, 20240401],
    ['2Y',  0.011, 'RF1', 20240101, 20240401],
    ['3Y',  0.014, 'RF1', 20240101, 20240401],
    ['4Y',  0.018, 'RF1', 20240101, 20240401],  # rupture volontaire
    ['5Y',  0.020, 'RF1', 20240101, 20240401]
]

df = pd.DataFrame(data, columns=['pillars', 'shock', 'rf_struct_id', 'scenario_id', 'as_of_date'])

# === Conversion des pillars en années ===
def convert_pillar(p):
    if 'Y' in p:
        return int(p.replace('Y', ''))
    elif 'M' in p:
        return int(p.replace('M', '')) / 12
    return np.nan

df['pillar_num'] = df['pillars'].apply(convert_pillar)

# === Application CUSUM sur les résidus spline ===
results = []
grouped = df.groupby(['rf_struct_id', 'scenario_id', 'as_of_date'])

for name, group in grouped:
    group = group.dropna(subset=['pillar_num']).sort_values('pillar_num')
    if len(group) < 4:
        continue

    try:
        x = group['pillar_num'].values
        y = group['shock'].values

        # Spline lissée plus stable
        spline = UnivariateSpline(x, y, s=1e-4)
        fitted = spline(x)
        residuals = y - fitted

        # Calcul du CUSUM
        mean_res = residuals.mean()
        cusum = np.cumsum(residuals - mean_res)

        # Sécurité : éviter division par 0
        std_cusum = np.std(cusum)
        if std_cusum == 0:
            z_scores = np.zeros_like(cusum)
        else:
            z_scores = (cusum - cusum.mean()) / std_cusum

        group = group.copy()
        group['residual'] = residuals
        group['cusum'] = cusum
        group['cusum_z'] = z_scores
        group['is_structural_break'] = np.abs(z_scores) > 2  # seuil de rupture

        results.append(group)

    except Exception as e:
        print(f"Erreur CUSUM sur {name}: {e}")
        continue

df_cusum = pd.concat(results, ignore_index=True)

# === Résultat : ruptures structurelles détectées ===
print("=== Ruptures détectées par CUSUM ===")
print(df_cusum[df_cusum['is_structural_break']])
