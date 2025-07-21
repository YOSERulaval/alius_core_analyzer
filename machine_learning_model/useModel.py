from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2, pandas as pd
from detectron2.utils.visualizer import Visualizer, ColorMode
from sympy.printing.pytorch import torch


class ModelPredictor:
    def __init__(self, model_dir, confidence_thresh=0.8, num_classes=1): # TODO num_classes n'est pas dynamique ? À vérifier
        # ----------------------------------Ne pas modifier cette partie-----------------------------------------------#
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes # Nombre de classes annotées
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh # Note de 'passage' pour afficher une classe prédite

        self.weights_path = os.path.join(model_dir, "model_final.pth")
        print("weights path is ", self.weights_path)
        self.cfg.MODEL.WEIGHTS = self.weights_path

        self.pixel_to_meter_ratio = 0.0

        self.predictor = DefaultPredictor(self.cfg) # type de prédicteur, via Detectron2
        # -------------------------------------------------------------------------------------------------------------#

        # La mémoire CUDA allouée est décidée par le modèle (mask_rcnn_R_50, par exemple) TODO : tester la stabilité avec d'autres modèles
        # À des fins de test seulement
        allocated_memory_bytes = torch.cuda.memory_allocated()
        allocated_memory_gigabytes = allocated_memory_bytes / (1024 ** 3)
        print(f"Mémoire CUDA allouée: {allocated_memory_gigabytes:.3f} GB")

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

        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu

        boxes = instances.pred_boxes.tensor.numpy()
        print("boxes: ", boxes)
        masks = instances.pred_masks.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        class_area = {class_name: 0 for class_name in class_names}

        for mask, class_id in zip(masks, classes):
            class_name = class_names[class_id]
            area_pixels = np.sum(mask)
            class_area[class_name] += area_pixels

        # Visualisation du résultat
        v = Visualizer(img[:, :, ::-1], scale=0.3, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances.to("cpu")) # image dans buffer cpu, peut être qu'on peut générer dans GPU ?
        cv2.imshow(img_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return class_names, class_area, img_name

    # Méthodes s'occupant de l'analyse, pourrait être implémenté dynamiquement selon les inputs du user (P3)
    def analyze_example(self, img_path):
        """
        Auteur : <NAME> \n
        Dernière MAJ : <DATE jj-mm-aaaa> \n
        Description : <DESC> \n
        :param img_path: Chemin vers l'image à analyser
        :return: Appel de __run_prediction
        """
        print("Ceci est un test")
        class_names = ["core", "joint", "broken_core"]
        #self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
        #print("number of classes : ", len(class_names))
        return self.__run_prediction(img_path, class_names)

    def analyze_joints(self, img_path):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 21-07-2025 \n
        Description : Méthode simpliste d'analyse de joint \n
        :param img_path: Chemin vers l'image à analyser
        :return: La position relative des joints, en ordre
        """
        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu

        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        class_names = ["core", "joint", "broken_core"]
        core_runs = [] # Liste de la classe parent
        joints = [] # Liste de la classe enfant

        for class_id, bbox, confidence in zip(classes, boxes, scores):
            if class_names[class_id] == "core":
                core_runs.append(CoreRun(bbox, confidence)) # core_runs → [CoreRun1, CoreRun2, ...]
            elif class_names[class_id] == "joint":
                joints.append(Joint(bbox, confidence)) # joints → [Joint1, Joint2, ...]

        for joint in joints:
            joint_center = joint.center() # on centre tous les objets JointSample
            for core_run in core_runs:
                if bbox_contains_point(core_run.bbox, joint_center):
                    core_run.add_joint(joint) # Si un joint est à l'intérieur d'une bbox on l'ajoute à la liste de joint de cet objet CoreRun

        """
        Important:
        Detectron2 tri les données de prédictions en ordre de taux de confiance, il faut donc faire un tri manuel avant 
        l'export vers Excel
        """
        core_runs_sorted = sort_cores(core_runs) # Tri des cores de haut en bas et gauche à droite

        cumulative_px = 0 # On commence l'analyse à zéro. La vraie profondeur est ajouté via Excel après

        self.pixel_to_meter_ratio = pixel_to_meter(core_runs_sorted, 0.06) # TODO pouvoir donner le choix entre HQ, NQ, HQ3, etc

        for c_idx, core in enumerate(core_runs_sorted):
            x1, _, x2, _ = core.bbox # on change seulement les valeurs en x de la bbox
            length_px = x2 - x1

            # Attributs en pixels
            core.x_start_px = cumulative_px
            core.x_end_px = cumulative_px + length_px
            core.length_px = length_px

            # Attributs en mètres, conversion selon le ratio calculé par 'pixel_to_meter_ratio'
            core.start_m = cumulative_px * self.pixel_to_meter_ratio
            core.end_m = core.x_end_px * self.pixel_to_meter_ratio
            core.length_m = length_px * self.pixel_to_meter_ratio

            if core.joints:
                # Si un joint est détecté dans un core, on MAJ ses attributs
                for j_idx, joint in enumerate(core.joints, start=0):
                    joint_pos_rel = cumulative_px + joint.center_x() # on conserve la distance relative accumulée
                    joint.distance_px = joint_pos_rel - core.x_start_px # position relative du joint selon le début de la run, en pixels
                    joint.distance_m = abs(joint.distance_px * self.pixel_to_meter_ratio) # position relative du joint selon le début de la run, en mètres

            cumulative_px += length_px
        self.export_core_run_to_excel(core_runs_sorted, os.path.basename(img_path)) # export vers excel

        return core_runs_sorted # au besoin

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
            "Analyze core dataset (joints, BC, LC, etc)": self.analyze_joints,
            "Analyze and crop lab core sample": self.analyze_crop_sample,
            "Test": self.analyze_example,
            # etc ...
        }

    @staticmethod
    def export_core_run_to_excel(core_runs, img_name):
        all_data = []

        for idx, core_run in enumerate(core_runs, start=1):
            core_id = f"Core_{idx}"
            core_run_data = {
                "core_run_id": core_id,
                "core_confidence": core_run.confidence,
                # "x_start_px": core_run.x_start_px,
                # "x_end_px": core_run.x_end_px,
                # "length_px": core_run.length_px,
                "from": core_run.start_m,
                "to": core_run.end_m,
                "run_length": core_run.length_m,
            }

            if core_run.joints:
                for joint_idx, joint in enumerate(core_run.joints, start=1):
                    joint_data = {
                        **core_run_data,
                        "joint_id": f"{core_id}_Joint_{joint_idx}",
                        "joint_x_position": joint.distance_m, # Après
                        #"joint_x_position": abs(joint.center()[0] # Avant
                        "joint_confidence": joint.confidence
                    }
                    all_data.append(joint_data)
            else:
                joint_data = {
                    **core_run_data,
                    "joint_id": None,
                    "joint_x_center": None,
                    "joint_confidence": None
                }
                all_data.append(joint_data)

        # Créer le DataFrame final
        df = pd.DataFrame(all_data)

        # Exporter vers Excel sur une seule feuille
        excel_path = f"{img_name}_core_runs.xlsx"
        df.to_excel(excel_path, sheet_name="Core_and_Joints", index=False)

        print(f"Données exportées vers : {excel_path}")

class CoreRun:
    """
    S'occupe des attributs d'analyse à l'intérieur d'un core (joint, broken core, lost core et faille)
    """
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.joints = [] # Liste vide qui accepte des listes de joints. Ex: pour core_1: [[coord. joint_1] []]

        # Caractéristiques de la run
        # en pixels
        self.x_start_px = None
        self.x_end_px = None
        self.length_px = None

        # en mètres
        self.start_m = None
        self.end_m = None
        self.length_m = None

    def add_joint(self, joint):
        """
        Associe un joint au core actuel (via self.joints).
        :param joint: liste contenant les coordonnées d'un joint, via l'instant créée de la classe 'Joint'.
        """
        self.joints.append(joint)

class Joint:
    """
    Gère les attributs d'un joint (point milieu, point centre, boundary box, niveau de confiance).
    """
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence

        # en pixels
        self.distance_px = None

        # en mètres
        self.distance_m = None

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def center_x(self):
        """
        :return: la coordonnée en x du point milieu d'un joint
        """
        x1, _, x2, _ = self.bbox
        pos_x = (x1 + x2) / 2
        return pos_x

# Fonctions globales, ne pas modifier
def bbox_contains_point(bbox, point):
    """
    Vérifie si un point (x, y) se trouve à l'intérieur d'une bbox [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox # Délimitation de la bbox
    x, y = point # Point centre du joint (ou d'un autre attribut)
    return x1 <= x <= x2 and y1 <= y <= y2 # return True si le point est à l'intérieur de la bbox, alors on sait que le joint (ou l'attribut) appartient à cette 'run' (core)

def sort_cores(core_runs):
    """
    :param core_runs: Accepte une liste de liste, ex: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...] qui représente tous
    les bbox des cores
    :return: Une liste de liste trié en fonction de y1 d'abord suivi de x1.
    """
    return sorted(core_runs, key=lambda core: (core.bbox[1], core.bbox[0]))

def pixel_to_meter(core_runs_sorted, core_diameter):
    """
    Calcul la hauteur (y2-y1) de TOUS les cores dans le dataset et fait la moyenne de cette hauteur.
    :param core_runs_sorted: Liste des cores triés.
    :param core_diameter: Diamètre du core à analyser (NQ, HQ, etc). Utiliser comme mesure de référence
    :return: Le ratio de conversion pixel<->mètre
    """
    d_core = core_diameter
    heights = [core.bbox[3] - core.bbox[1] for core in core_runs_sorted]
    d_average = sum(heights) / len(heights) # moyenne

    return d_core / d_average