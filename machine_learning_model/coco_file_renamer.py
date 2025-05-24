"""
Comme avec 'img_renamer', ce code renomme le nom des images enregistrer dans un projet d'annotation.

-Yohan
"""

import os
import json
import tkinter as tk
from tkinter import filedialog

def clean_name(name: str) -> str:
    """
    Remplace les points dans le nom de base par des underscores,
    en conservant l'extension intacte.
    """
    stem, ext = os.path.splitext(name)
    new_stem = stem.replace('.', '_').replace(' ', '_').replace('-', '_')
    return new_stem + ext

def rename_in_json(json_path: str, output_path: str):
    # Charge le JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Traite le dictionnaire d'images
    old_meta = data.get('_via_img_metadata', {})
    new_meta = {}
    new_id_list = []

    for old_key, meta in old_meta.items():
        old_fname = meta['filename']
        size = meta['size']  # taille en octets, inclus dans la clé originale :contentReference[oaicite:0]{index=0}
        new_fname = clean_name(old_fname)

        # Nouvelle clé = nouveau nom + taille
        new_key = f"{new_fname}{size}"

        # Met à jour la valeur filename
        meta['filename'] = new_fname

        # Conserve le reste (regions, file_attributes…)
        new_meta[new_key] = meta
        new_id_list.append(new_key)

    # Remplace dans la structure globale
    data['_via_img_metadata'] = new_meta
    data['_via_image_id_list'] = new_id_list

    # Sauvegarde
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✔ JSON mis à jour et enregistré sous : {output_path}")

if __name__ == "__main__":
    # Boîte de dialogue pour choisir le fichier JSON
    root = tk.Tk()
    root.withdraw()
    json_file = filedialog.askopenfilename(
        title="Sélectionnez votre fichier .json VIA/COCO",
        filetypes=[("JSON files", "*.json")]
    )
    root.destroy()

    if not json_file:
        print("Aucun fichier sélectionné, sortie.")
        exit(0)

    # Définit le chemin de sortie
    base = os.path.basename(json_file)
    out_file = os.path.join(
        os.path.dirname(json_file),
        f"renamed_{base}"
    )

    rename_in_json(json_file, out_file)
