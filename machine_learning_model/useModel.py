from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2, pandas as pd
from detectron2.utils.visualizer import Visualizer, ColorMode
from sympy.printing.pytorch import torch


class ModelPredictor:
    def __init__(self, model_dir, confidence_thresh=0.9, num_classes=1):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes # Nombre de classes annotées
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh # Note de 'passage' pour afficher une classe prédite

        self.weights_path = os.path.join(model_dir, "model_final.pth")
        print("weights path is ", self.weights_path)
        self.cfg.MODEL.WEIGHTS = self.weights_path

        self.predictor = DefaultPredictor(self.cfg)

    def __run_prediction(self, img_path, class_names):
        """
        Méthode privée encapsulant tous les paramètres communs d'une analyse d'image standard
        :param img_path: Chemin vers l'image à analyser
        :param class_names: Nom de la ou les classes présentes dans l'annotation
        :return: class_names, class_area, img_name
        """
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        outputs = self.predictor(img)

        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        class_area = {class_name: 0 for class_name in class_names}

        for mask, class_id in zip(masks, classes):
            class_name = class_names[class_id]
            area_pixels = np.sum(mask)
            class_area[class_name] += area_pixels

        # Visualisation du résultat
        v = Visualizer(img[:, :, ::-1], scale=0.3, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances.to("cpu"))
        cv2.imshow(img_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return class_names, class_area, img_name

    def export_joints_to_excel(self, joint_position, img_name):
        df = pd.DataFrame(joint_position) # Creer un tableau des positions des joints
        excel_path = f"{img_name}_joints_positions.xlsx"
        df.to_excel(excel_path, index=False)

        print(f"Positions des joints exportées vers : {excel_path}")

    # Méthodes s'occupant de l'analyse, pourrait être implémenté dynamiquement selon les inputs du user (P3)
    def analyze_example(self, img_path):
        """
        <DESC> \n
        Auteur : <NAME>
        :param img_path: Chemin vers l'image à analyser
        :return: Appel de __run_prediction
        """
        print("Ceci est un test")
        class_names = ["core"]
        return self.__run_prediction(img_path, class_names)

    def analyze_joints(self, img_path):
        """
        Méthode simpliste d'analyse de joint \n
        Auteur: Yohan Sercia
        :param img_path: Chemin vers l'image à analyser
        :return: La position relative des joints, en ordre
        """
        print("Analyzing joints")
        class_names = ["joint"]
        class_names, class_area, img_name = self.__run_prediction(img_path, class_names)

        img = cv2.imread(img_path)
        outputs = self.predictor(img)

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()

        joint_positions = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            joint_positions.append({"x_center": x_center, "y_center": y_center})

        joint_positions_sorted = sorted(
            joint_positions,
            key=lambda pos: (pos["y_center"], pos["x_center"])
        )

        self.export_joints_to_excel(joint_positions_sorted, img_name)  # Export vers Excel

        return joint_positions_sorted

    def analyze_crop_sample(self, img_path):
        """
        Méthode simpliste d'analyse de crop sample \n
        Auteur: Yohan Sercia
        :param img_path: Chemin vers l'image à analyser
        :return: Lance l'analyse de crop sample via __run_prediction
        """
        print("Analyzing core sample")
        class_names = ["core"]
        return self.__run_prediction(img_path, class_names)

    def get_methods(self):
        """
        Accesseur des méthodes d'analyses, ajoutez une référence à votre méthode d'analyse ici afin de pouvoir l'utiliser dans l'interface.
        :return: Retourne un dictionnaire avec les méthodes d'analyses en clés
        """
        return {
            "Analyze joints": self.analyze_joints,
            "Analyze core sample": self.analyze_crop_sample,
            "Test": self.analyze_example
        }

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