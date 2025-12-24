"""
Auteur : Yohan Sercia \n
Dernière MAJ : 25-09-2025 \n
Description : Création d'un graphe de comparaison entre le trou loggé et la prédiction. \n
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === FICHIERS & PARAMS ===
excel_file_real = "NOV-002 Core logging YS - Courvan 13-07-2025.xlsx"  # excel loggé manuellement
excel_file_predicted = "Test avec modèle Charlie/COG-25-12 (predicted).xlsx"  # excel prédit par le modèle

geotech_sheet = "Geotech"            # onglet avec les runs
disc_sheet    = "Discontinuities"    # onglet avec les discontinuités
hole_of_interest = "COG-25-12"

# noms de colonnes
hole_col = "Hole ID"
from_col = "From (m)"
to_col = "To (m)"
disc_depth_col = "Depth (m)" # profondeur des discontinuités

def hole_cumulatinator(
        xlsx_file_path,
        geotech_sheet,
        disc_sheet,
        header_sheet,
        hole_col,
        disc_depth_col,
        hole_of_interest=''
):
    # === LECTURE DES DONNÉES ===
    # ---Géotech---
    geo = pd.read_excel(xlsx_file_path, sheet_name=geotech_sheet, header=header_sheet)
    if hole_of_interest.strip() != '':
        geo = geo[geo[hole_col].astype(str).str.strip() == hole_of_interest].copy()

    initial_run = geo["From (m)"].min()
    last_run = geo["To (m)"].max()

    if geo.empty:
        raise ValueError(f"Aucun run trouvé pour le trou '{hole_of_interest}' dans l'onglet {geotech_sheet}.")

    hole_extremities = [initial_run, last_run]

    # ---Discontinuités---
    disc = pd.read_excel(xlsx_file_path, sheet_name=disc_sheet, header=header_sheet)
    if hole_of_interest.strip() != '':
        disc = disc[disc[hole_col].astype(str).str.strip() == hole_of_interest].copy()

    if disc_depth_col not in disc.columns:
        raise KeyError(f"La colonne '{disc_depth_col}' est introuvable dans l'onglet {disc_sheet}. "
                       f"Colonnes trouvées: {list(disc.columns)}")

    disc_depths = pd.to_numeric(disc[disc_depth_col], errors="coerce").dropna().to_numpy()

    return hole_extremities, disc_depths

hole_depth_real, disc_depth_real = hole_cumulatinator( # Création du graphe du trou réel
    excel_file_real,
    geotech_sheet,
    disc_sheet,
    2,
    hole_col,
    disc_depth_col,
    hole_of_interest
)

hole_depth_predicted, disc_depth_predicted = hole_cumulatinator( # Création du graphe du trou prédit
    excel_file_predicted,
    geotech_sheet,
    disc_sheet,
    0,
    hole_col,
    disc_depth_col,
    hole_of_interest
)

# ===PLOT===
fig, ax = plt.subplots(figsize=(3.2, 8))

x_real = -0.5
x_pred = +0.5

# “puits” vertical à x=0
ax.plot([x_real, x_real], [hole_depth_real[0], hole_depth_real[1]], linewidth=8, color="steelblue", label="Réel")
ax.plot([x_pred, x_pred], [hole_depth_predicted[0], hole_depth_predicted[1]], linewidth=8, color="orange", label="Prédit")

# points de discontinuités à x=0 (léger décalage pour visibilité si tu veux)
ax.hlines(y=disc_depth_real, xmin=x_real-0.30, xmax=x_real+0.30, linewidth=1, color="steelblue", alpha=0.8)
ax.hlines(y=disc_depth_predicted, xmin=x_pred-0.30, xmax=x_pred+0.30, linewidth=1, color="orange", alpha=0.8)


ax.invert_yaxis()                 # profondeur vers le bas
ax.set_xlim(-1.0, 1.0)            # espace pour voir les deux puits
ax.set_xticks([])
ax.set_ylabel("Profondeur (m)")
ax.set_title(hole_of_interest)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f"{hole_of_interest}_discontinuities.png", dpi=200, bbox_inches="tight")
plt.show()