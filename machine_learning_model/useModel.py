from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2
from detectron2.utils.visualizer import Visualizer, ColorMode

class ModelPredictor:
    def __init__(self, device='cpu', num_classes=3, confidence_thresh=0.7):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        current_dir = os.path.dirname(os.path.abspath(__file__)) # On prend le chemin absolu du output
        self.cfg.MODEL.WEIGHTS = os.path.join(current_dir, self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = ["cz", "core_empty_or_broken", "core_full"]

    def analyze_image(self, image_path):
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        outputs = self.predictor(img)

        # Extraction des classes utiles pour nos calculs
        instances = outputs["instances"]
        masks = instances.pred_masks.cpu().numpy() # Ce que la machine a prédit
        classes = instances.pred_classes.cpu().numpy() # Ce que nous avons défini comme classe

        # Calculs pour le ratio d'aire, pour tests seulement
        class_names = ["cz", "core_empty_or_broken", "core_full"]
        areas = {"cz": 0, "core_empty_or_broken": 0, "core_full": 0}

        # Calcul de l'aire en pixel pour chacune des classes
        for mask, class_id in zip(masks, classes):
            class_name = class_names[class_id]
            area_pixels = np.sum(mask)
            areas[class_name] += area_pixels

        ratio = (areas["core_empty_or_broken"] / areas["core_full"]) * 100

        # Pour visualiser la prédiction faite sur l'image test (avant le calcul)
        v = Visualizer(img[:, :, ::-1], scale=0.35, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(image_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return class_names, areas, ratio, image_name


"""
# Configuration complète de Detectron2 (Mask-RCNN)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # même nombre de classes que lors de l'entraînement
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


# Pour visualiser la prédiction faite sur l'image test (avant le calcul)
im = cv2.imread(image_path)
v = Visualizer(im[:, :, ::-1],
                   scale=0.2,
                   instance_mode=ColorMode.IMAGE_BW
                   )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow(image_path, out.get_image()[:, :, ::-1])
cv2.waitKey(0)


for mask, class_id in zip(masks, classes):
    class_name = class_names[class_id]
    area_pixels = np.sum(mask)
    areas[class_name] += area_pixels

print("Aires calculées (pixels):", areas, " (pixels)")

ratio = (areas["core_empty_or_broken"] / areas["core_full"]) * 100
print(f"Rapport broken/full : <{ratio:.2f}%> pour l'image : <{image_name}>.")
"""