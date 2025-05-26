from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os, json, cv2, random

# Configuration complète de Detectron2 (Mask-RCNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # très important : même nombre de classes que lors de l'entraînement
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # seuil de confiance
cfg.MODEL.DEVICE = 'cpu'  # ou 'cuda' si tu utilises un GPU

predictor = DefaultPredictor(cfg)

def path_picker_image():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Choisir une image à mettre dans le modèle",
                                      filetypes=(("image files", ".jpg"),))
    root.destroy()
    if not path:
        print("Aucun fichier sélectionné, sortie.")
        exit(0)
    return path

image_path = path_picker_image()
image_name = os.path.basename(image_path)
img = cv2.imread(image_path)
outputs = predictor(img)

instances = outputs["instances"]
masks = instances.pred_masks.cpu().numpy()
classes = instances.pred_classes.cpu().numpy()

class_names = ["cz", "core_empty_or_broken", "core_full"]

areas = {"cz": 0, "core_empty_or_broken": 0, "core_full": 0}

for mask, class_id in zip(masks, classes):
    class_name = class_names[class_id]
    area_pixels = np.sum(mask)
    areas[class_name] += area_pixels

print("Aires calculées (pixels):", areas, " (pixels)")

#rqd = (areas["core_full"] / (areas["core_full"] + areas["core_empty_or_broken"])) * 100
ratio = (areas["core_empty_or_broken"] / areas["core_full"]) * 100
print(f"Rapport broken/full : <{ratio:.2f}%> pour l'image : <{image_name}>.")
