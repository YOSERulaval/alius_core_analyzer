from sympy.physics.units import length

def export_core_runs_to_excel(core_runs, output_name, output_path):
    """
    Auteur : ChatGPT + Yohan Sercia
    Dernière MAJ : 21-09-2025
    Description : Exporte dans un fichier Excel avec deux onglets :
      - Core_Runs  : info des runs (sans colonnes de fractures), dédupliqué
      - Fractures  : mêmes colonnes que l'export actuel (avec fractures)
    """
    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.styles import numbers
    import numpy as np
    import os

    # -------- 1) Construire les deux tableaux --------
    fractures_rows = []
    core_rows = []

    TYPES_WITH_LENGTH = {"broken_core", "fault"}  # largeur des discon seulement pour ceux là

    for idx, core in enumerate(core_runs):
        # L'identifiant de run reste indexé dans l’ordre d’apparition par image :
        core_id = f"{core.img_name}_Core_{idx}"

        core_base = {
            "Hole ID": "COG-25-12", # TODO mettre dynamique
            "img_name": core.img_name,
            "core_run_id": core_id,
            "From (m)": core.start_m,
            "To (m)": core.end_m,
            "Run (m)": core.length_m,
            "RQD (m)": getattr(core, "total_rqd_length", None),
            "RQD (%)": getattr(core, "core_rqd", None),
        }

        #  ---Feuille "Geotech"---
        core_rows.append(core_base.copy())

        # ---Feuille "Discontinuités"---
        if core.fractures:
            for f_idx, f in enumerate(core.fractures):
                row = {
                    **core_base,
                    " ": " ",  # espacement
                    "fracture_id": f"{core_id}_{f.fracture_type}_{f_idx}",
                    "Type": f.fracture_type,
                    "Depth (m)": core.start_m + f.distance_m,
                    "Length (m)": f.length_m if f.fracture_type in TYPES_WITH_LENGTH else None,
                    "": " ",  # espacement
                }
                fractures_rows.append(row)

        else:
            row = {
                **core_base,
                " ": " ",
                "fracture_id": None,
                "Type": None,
                "Depth (m)": None,
                "Length (m)": None,
                "": " "
            }
            fractures_rows.append(row)

    cores_df = pd.DataFrame(core_rows)
    fractures_df = pd.DataFrame(fractures_rows)

    # -------- 2.1) Dédupe des runs (Core_Runs) --------
    # On arrondit les flottants utilisés pour la clé de dédup (prévient les petits écarts binaires).
    for col in ["From (m)", "To (m)", "Run (m)", "RQD (m)", "RQD (%)"]:
        if col in cores_df.columns:
            cores_df[col] = cores_df[col].astype(float).round(6)

    dedup_key = ["Hole ID", "img_name", "From (m)", "To (m)", "Run (m)", "RQD (m)", "RQD (%)"]
    cores_df = cores_df.drop_duplicates(subset=dedup_key, keep="first").reset_index(drop=True)

    # -------- 2.2) Colonnes pertinentes pour les fractures --------
    fracture_cols = [
        "Hole ID",
        "img_name",
        "fracture_id",
        "Type",
        "Depth (m)",
        "Length (m)",
    ]

    # Re-indexe pour obtenir exactement ces colonnes (celles manquantes seront créées vides)
    fractures_df = fractures_df.reindex(columns=fracture_cols)

    # Arrondi léger sur les colonnes numériques
    for col in ["Depth (m)", "Length (m)"]:
        if col in fractures_df.columns:
            fractures_df[col] = pd.to_numeric(fractures_df[col], errors="coerce").round(6)

    # -------- 3) Sauvegarde Excel --------
    excel_path = os.path.join(output_path, f"{output_name}_cores_fractures.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        cores_df.to_excel(writer, sheet_name="Geotech", index=False)
        fractures_df.to_excel(writer, sheet_name="Discontinuities", index=False)

    # -------- 4) Formatage openpyxl --------
    wb = load_workbook(excel_path)

    # -- Core_Runs
    ws_core = wb["Geotech"]
    core_col_widths = {
        'A': 20,  # Hole ID
        'B': 30,  # img_name
        'C': 35,  # core_run_id
        'D': 10,  # from
        'E': 10,  # to
        'F': 12,  # run_length
        'G': 12,  # rqd_length
        'H': 10,  # RQD (%)
    }
    for col, w in core_col_widths.items():
        ws_core.column_dimensions[col].width = w

    # Format numérique : 2 décimales à partir de la colonne C (numériques)
    for row in ws_core.iter_rows(min_row=2, max_row=ws_core.max_row, min_col=3, max_col=ws_core.max_column):
        for cell in row:
            if isinstance(cell.value, (float, int)) and cell.value is not None:
                cell.number_format = '0.00'

    # -- Fractures
    ws_frac = wb["Discontinuities"]
    frac_col_widths = {
        'A': 20,  # Hole ID
        'B': 30,  # img_name
        'C': 40,  # fracture_id
        'D': 20,  # type
        'E': 12,  # position
        'F': 12,  # length
    }
    for col, w in frac_col_widths.items():
        ws_frac.column_dimensions[col].width = w

    # 2 décimales pour les colonnes numériques (à partir de C)
    for row in ws_frac.iter_rows(min_row=2, max_row=ws_frac.max_row, min_col=3, max_col=ws_frac.max_column):
        for cell in row:
            if isinstance(cell.value, (float, int)) and cell.value is not None:
                cell.number_format = '0.00'

    wb.save(excel_path)
    print(f"Données exportées vers : {excel_path}")
