"""
Alius Consultation Minière inc.
====================================
CRÉÉ PAR:       Yohan Sercia
CRÉE LE:        17-12-2025
------------------------------------
PROJET:         SP Formatter
DESCRIPTION:    Formatage des photos de labs pour les échantillons BTS, UCS et TCS.
                Comprends: Retirer le fond de l'image + rotation de celle-ci au besoin (si à l'envers).
------------------------------------
DERNIÈRE MAJ:   17-12-2025
====================================
"""
import os, cv2

from sources_files import methodLinker as mlinker
from sources_files import utilitiesMethods as utils

output_dir = "project_output"

class SPFormatterAPT(mlinker.Method):
    name = "APT - SP Formatter (YS)"  # (SamplePhoto) Formatter

    def run(self, img_path: str):
        utils.clean_cache() # Toujours commencer par s'assurer que la cache est vide avant

        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)

        return self._crop_core_sample(img_path)


    def _crop_core_sample(self, img_path):
        print("[SPFormatter] Analyzing core sample")

        img = cv2.imread(img_path)

        outputs = self.model.predictor(img) # Appel propre au ModelPredictor
        instances = outputs["instances"].to("cpu")

        if len(instances.pred_masks) == 0:
            print("[SPFormatter] Aucun sample détecté.")
            return None

        mask = instances.pred_masks[0].numpy().astype("uint8")
        y, x = mask.nonzero()
        cropped_img = img[y.min():y.max(), x.min():x.max()]

        filename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{filename}_cropped.png")

        cv2.imwrite(out_path, cropped_img)
        print(f"[SPFormatter] Image sauvegardée : {out_path}")

        return out_path