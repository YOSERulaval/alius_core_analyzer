import os
import json
import tkinter as tk
from tkinter import filedialog

def build_sequential_map(directory, prefix="img", start_index=0):
    files = sorted(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
    mapping = {}
    for idx, fname in enumerate(files, start=start_index):
        ext = os.path.splitext(fname)[1]  # .jpg, .png, etc.
        mapping[fname] = f"{prefix}_{idx}{ext}"
    return mapping

def batch_rename_files(directory, mapping, dry_run=True):
    for old, new in mapping.items():
        old_path = os.path.join(directory, old)
        new_path = os.path.join(directory, new)
        if old_path == new_path:
            continue
        if dry_run:
            print(f"[DRY-RUN] {old} → {new}")
        else:
            print(f"Renommage : {old} → {new}")
            os.rename(old_path, new_path)

def update_via_json(json_path, mapping, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    old_meta = data.get('_via_img_metadata', {})
    new_meta = {}
    new_id_list = []

    for old_key, meta in old_meta.items():
        old_fname = meta['filename']
        size = meta['size']
        if old_fname not in mapping:
            print(f"⚠️  Fichier non trouvé dans le mapping : {old_fname}, on saute.")
            continue
        new_fname = mapping[old_fname]
        # met à jour le champ filename
        meta['filename'] = new_fname
        # nouvelle clé = nouveau nom + taille (comme le format VIA)
        new_key = f"{new_fname}{size}"
        new_meta[new_key] = meta
        new_id_list.append(new_key)

    data['_via_img_metadata']   = new_meta
    data['_via_image_id_list']  = new_id_list

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✔ JSON mis à jour et sauvegardé sous : {output_path}")

if __name__ == "__main__":
    # Initialisation Tkinter pour les dialogues
    root = tk.Tk()
    root.withdraw()

    # 1. Choix du dossier d'images
    img_dir = filedialog.askdirectory(title="Sélectionnez le dossier d'images")
    if not img_dir:
        print("Aucun dossier sélectionné, fin.")
        exit(1)

    # 2. Choix du fichier JSON VIA
    json_file = filedialog.askopenfilename(
        title="Sélectionnez le fichier JSON VIA/COCO",
        filetypes=[("Fichiers JSON", "*.json")]
    )
    root.destroy()
    if not json_file:
        print("Aucun fichier JSON sélectionné, fin.")
        exit(1)

    # 3. Construction du mapping
    mapping = build_sequential_map(img_dir, prefix="img", start_index=1)

    # 4. Dry-run du renommage
    print("=== DRY-RUN : vérification des renommages ===")
    batch_rename_files(img_dir, mapping, dry_run=False)
    print("\nSi tout vous convient, ré-exécutez en passant dry_run=False dans batch_rename_files")

    # 5. (Optionnel) Pour appliquer réellement : décommentez la ligne suivante
    # batch_rename_files(img_dir, mapping, dry_run=False)

    # 6. Mise à jour du JSON
    base = os.path.basename(json_file)
    out_json = os.path.join(os.path.dirname(json_file), f"renamed_{base}")
    update_via_json(json_file, mapping, out_json)
