"""
Fichier pour modéliser un trou géotechnique loggué manuellement vs le même trou loggué par l'IA
"""

import pandas as pd
import matplotlib.pyplot as plt

excel_file_real = "NOV-002 Core logging YS - Courvan 13-07-2025.xlsx"   # le fichier .xlsx est dans le même dossier que ce script
excel_file_pred = "COG-25-10 (predicted).xlsx"

sheet_name = ["Geotech", "Discontinuities"]        # ou un index, ex. 0
hole_of_interest = "COG-25-10"

df = pd.read_excel(excel_file_real, sheet_name=sheet_name[0], header=2)

# colonnes d'intérêts
run_col  = "Run (m)"
discontinuities_col = "Discontinuities (m)"

d = df[df["Hole ID"].astype(str).str.strip() == hole_of_interest] # On garde seulement les données pertinentes à notre trou

