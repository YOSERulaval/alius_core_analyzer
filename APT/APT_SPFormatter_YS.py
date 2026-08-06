"""
Alius Consultation Minière
====================================
CRÉÉ PAR:       Yohan Sercia
CRÉE LE:        17-12-2025
------------------------------------
PROJET:         SP Formatter
DESCRIPTION:    Formatage des photos de labs pour les échantillons BTS, UCS et TCS.
                Comprends: Retirer le fond de l'image.
------------------------------------
DERNIÈRE MAJ:   22-04-2026
====================================
"""
import os, cv2, numpy as np
import glob

from src import methodLinker as mlinker
from src import utilitiesMethods as utils

class SPFormatterAPT(mlinker.Method):
    name = "APT - SPFormatter (YS)"  # (SamplePhoto) Formatter

    def run(self, parent_folder_path: str):
        print("[SPFormatter] Analyzing core sample")
        utils.clean_cache() # Toujours commencer par s'assurer que la cache est vide avant

        for root, dirs, files in os.walk(parent_folder_path):
            img_files = (
                    glob.glob(os.path.join(root, "*.jpg")) +
                    glob.glob(os.path.join(root, "*.jpeg")) +
                    glob.glob(os.path.join(root, "*.png"))
            )

            if not img_files:
                print("Aucune image trouvée dans le dossier spécifié.")
                continue

            output_dir = os.path.join(root, "SPFormatter_output")
            os.makedirs(output_dir, exist_ok=True)

            for idx, img_path in enumerate(img_files):
                self._bg_core_sample(img_path, output_dir, idx, len(img_files))
        print(f"[SPFormatter] APT terminee !")

    @staticmethod
    def _enforce_ratio(img, ratio):
        h, w = img.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - ratio) < 1e-3:
            return img  # déjà bon

        # Cas 1 : image trop large → ajouter padding vertical
        if current_ratio > ratio:
            new_h = int(w / ratio)
            pad = new_h - h

            pad_top = pad // 2
            pad_bottom = pad - pad_top

            result = cv2.copyMakeBorder(
                img,
                pad_top, pad_bottom, 0, 0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
            )

        # Cas 2 : image trop haute → ajouter padding horizontal
        else:
            new_w = int(h * ratio)
            pad = new_w - w

            pad_left = pad // 2
            pad_right = pad - pad_left

            result = cv2.copyMakeBorder(
                img,
                0, 0, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
            )

        return result

    def _bg_core_sample(self, img_path: str, output_dir: str,  idx: int, total: int) -> str | dict | None:
        """
        Lecture d'une seule image pour l'instant
        :param img_path:
        :return:
        """
        print(f"[SPFormatter] ({idx+1}/{total}) Analyse de ", os.path.basename(img_path))

        # Image et formatage en 1:2 (UCS/TCS) et 1:1 (BTS)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SPFormatter] Impossible de lire l'image : {img_path}")
            return None

        target_ratio: float = 1.0

        outputs = self.model.predictor(img) # Appel propre au ModelPredictor
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            print("[SPFormatter] Aucun objet détecté")
            detected_class_idx = -1  # ou une valeur spéciale
        else:
            detected_class_idx = int(instances.pred_classes[0])

        #detected_class_idx = int(instances.pred_classes.cpu().numpy()[0])

        # ['UCS&TCS', 'BTS']
        if detected_class_idx == 0: target_ratio = 0.5  # Cas 'UCS&TCS'
        elif detected_class_idx == 1: target_ratio = 1.0  # Cas 'BTS'

        if len(instances) == 0:
            return {
                "success": False,
                "reason": "no_detection"
            }

        mask = instances.pred_masks[0].numpy().astype("uint8") # 0 = fond, 1 = objet
        alpha = mask * 255

        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, alpha])

        img_formated = self._enforce_ratio(rgba, target_ratio)

        filename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{filename}.png")

        cv2.imwrite(out_path, img_formated)

        return out_path