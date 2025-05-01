import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import re
import warnings

warnings.filterwarnings("ignore")

# === Exemple de dataset (remplace par df = pd.read_csv(...)) ===
data = [
    ['1Y',  0.010, 'RF1', 20240101, 20240401],
    ['2Y',  0.011, 'RF1', 20240101, 20240401],
    ['3Y',  0.015, 'RF1', 20240101, 20240401],  # cassure volontaire
    ['4Y',  0.011, 'RF1', 20240101, 20240401],
    ['5Y',  0.010, 'RF1', 20240101, 20240401]
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

# === Détection de cassures locales via dérivée seconde ===
results = []
grouped = df.groupby(['rf_struct_id', 'scenario_id', 'as_of_date'])

for name, group in grouped:
    group = group.dropna(subset=['pillar_num']).sort_values('pillar_num')
    if len(group) < 4:
        continue

    x = group['pillar_num'].values
    y = group['shock'].values

    try:
        # Augmente légèrement s pour éviter les warnings
        spline = UnivariateSpline(x, y, s=1e-4)
        second_derivative = spline.derivative(n=2)(x)

        # Protéger le calcul du z-score
        if np.std(second_derivative) == 0:
            z_scores = np.zeros_like(second_derivative)
        else:
            z_scores = (second_derivative - np.mean(second_derivative)) / np.std(second_derivative)

        group = group.copy()
        group['f_second_deriv'] = second_derivative
        group['curvature_z'] = z_scores
        group['is_breakpoint'] = np.abs(z_scores) > 2  # seuil de cassure locale

        results.append(group)

    except Exception as e:
        print(f"Erreur spline dérivée seconde sur {name}: {e}")
        continue

df_breaks = pd.concat(results, ignore_index=True)

# === Résultat : cassures locales détectées ===
print("=== Cassures détectées par dérivée seconde ===")
print(df_breaks[df_breaks['is_breakpoint']])
