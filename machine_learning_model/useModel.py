from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2, glob, pandas as pd
from detectron2.utils.visualizer import Visualizer, ColorMode
from sympy.printing.pytorch import torch


class ModelPredictor:
    def __init__(self, model_dir, confidence_thresh=0.8, num_classes=5): # TODO num_classes n'est pas dynamique ? À vérifier
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

        #torch.cuda.empty_cache() # vider la cache avant chaque run
        # -------------------------------------------------------------------------------------------------------------#

        # La mémoire CUDA allouée est décidée par le modèle (mask_rcnn_R_50, par exemple) TODO : tester avec d'autres modèles
        # À des fins de test seulement
        allocated_memory_bytes = torch.cuda.memory_allocated()
        allocated_memory_gigabytes = allocated_memory_bytes / (1024 ** 3)

        cached_memory_bytes = torch.cuda.memory_reserved()
        cached_memory_gigabytes = cached_memory_bytes / (1024 ** 3)

        print(f"Mémoire CUDA allouée: {allocated_memory_gigabytes:.3f} GB")
        print(f"Mémoire CUDA cache: {cached_memory_gigabytes:.3f} GB")

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
        #print("boxes: ", boxes)
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
        # Exemple de docstring à mettre dans votre méthode:
        """
        Auteur : <NAME> \n
        Dernière MAJ : <jj-mm-aaaa> \n
        Description : <DESC> \n
        :param img_path: Chemin vers l'image à analyser
        :return: Appel de __run_prediction
        """
        print("Ceci est un test")
        clean_cache() # Libérer la mémoire du GPU AVANT de commencer une analyse.
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # !! IMPORTANT de mettre votre 'note de passage' !!
        class_names = ["core", "joint", "broken_core", "lost_core", "fault"] # Ajoutez toutes les classes présentes sur vos annotations

        return self.__run_prediction(img_path, class_names)

    def analyze_joints(self, img_path):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 25-07-2025 \n
        Description : Méthode simpliste d'analyse de joint \n
        :param cumulative_px:
        :param img_path: Chemin vers l'image à analyser
        :return: La position relative des joints, en ordre
        """
        clean_cache()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu

        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        class_names = ["core", "joint", "broken_core", "lost_core", "fault"]

        # Liste des cores (parent) et des fractures (enfant)
        core_runs = []
        fractures = []

        for class_id, bbox, confidence in zip(classes, boxes, scores):
            if class_names[class_id] == "core":
                core_runs.append(CoreRun(bbox, confidence))
            if class_names[class_id] in ["joint", "broken_core", "lost_core", "fault"]:
                fracture = Fracture(bbox, confidence, class_names[class_id])
                fractures.append(fracture)

        """
        Important:
        Detectron2 tri les données de prédictions en ordre décroissant de taux de confiance, il faut donc faire un tri manuel avant 
        l'export vers Excel
        """
        core_runs_sorted = sort_cores(core_runs) # Tri des cores de haut en bas et gauche à droite

        cumulative_px = 0 # On commence l'analyse à zéro. La vraie profondeur est ajouté via Excel après

        self.pixel_to_meter_ratio = pixel_to_meter(core_runs_sorted, 0.0635) # TODO pouvoir donner le choix entre HQ, NQ, HQ3 dans l'interface

        # Boucle principale qui calcul les mesures cumulées
        for core in core_runs_sorted:
            x1, _, x2, _ = core.bbox
            length_px = x2 - x1

            core.x_start_px = cumulative_px
            core.x_end_px = cumulative_px + length_px
            core.length_px = length_px

            core.start_m = cumulative_px * self.pixel_to_meter_ratio
            core.end_m = core.x_end_px * self.pixel_to_meter_ratio
            core.length_m = length_px * self.pixel_to_meter_ratio

            # Associer clairement chaque fracture à son core
            for fracture in fractures:
                if bbox_contains_point(core.bbox, fracture.center()):
                    fracture.compute_distances(core.bbox, self.pixel_to_meter_ratio)
                    core.add_fracture(fracture)  # méthode unique add_fracture()

            cumulative_px += length_px

        #self.export_core_run_to_excel(core_runs_sorted, os.path.basename(img_path)) # export vers excel
        return core_runs_sorted

    def analyze_crop_sample(self, img_path):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 21-07-2025 \n
        Méthode simpliste d'analyse de crop sample du lab \n
        :param img_path: Chemin vers l'image à analyser
        :return: Lance l'analyse de crop sample via __run_prediction
        """
        clean_cache()
        print("Analyzing core sample")
        class_names = ["core"]
        return self.__run_prediction(img_path, class_names)

    @staticmethod
    def export_core_run_to_excel(core_runs, img_name):
        """
        Pour une seule image
        :param core_runs:
        :param img_name:
        """
        all_data = []

        for idx, core in enumerate(core_runs):
            core_data = {
                "core_run_id": f"Core_{idx}",
                "from": core.start_m,
                "to": core.end_m,
                "run_length": core.length_m
            }

            if core.fractures:
                for f_idx, fracture in enumerate(core.fractures):
                    fracture_data = {
                        **core_data,
                        "fracture_id": f"{core_data['core_run_id']}_{fracture.fracture_type}_{f_idx}",
                        "fracture_type": fracture.fracture_type,
                        "position_m": fracture.distance_m,
                        "length_m": fracture.length_m
                    }
                    all_data.append(fracture_data)
            else:
                no_fracture_data = {
                    **core_data,
                    "fracture_id": None,
                    "fracture_type": None,
                    "position_m": None,
                    "length_m": None
                }
                all_data.append(no_fracture_data)

        df = pd.DataFrame(all_data)
        excel_path = f"{img_name}_core_fractures.xlsx"
        df.to_excel(excel_path, sheet_name="Fractures", index=False)
        print(f"Données exportées vers : {excel_path}")

    @staticmethod
    def export_multiple_core_runs_to_excel(core_runs, output_name):
        """
        Pour plusieurs images consécutives
        :param core_runs:
        :param output_name:
        """
        all_data = []

        for idx, core in enumerate(core_runs):
            core_data = {
                "img_name": core.img_name,
                "core_run_id": f"{core.img_name}_Core_{idx}",
                "from": core.start_m,
                "to": core.end_m,
                "run_length": core.length_m
            }

            if core.fractures:
                for f_idx, fracture in enumerate(core.fractures):
                    fracture_data = {
                        **core_data,
                        "fracture_id": f"{core_data['core_run_id']}_{fracture.fracture_type}_{f_idx}",
                        "fracture_type": fracture.fracture_type,
                        "position_m": fracture.distance_m,
                        "length_m": fracture.length_m
                    }
                    all_data.append(fracture_data)
            else:
                no_fracture_data = {
                    **core_data,
                    "fracture_id": None,
                    "fracture_type": None,
                    "position_m": None,
                    "length_m": None
                }
                all_data.append(no_fracture_data)

        df = pd.DataFrame(all_data)
        excel_path = f"{output_name}_cores_fractures.xlsx"
        df.to_excel(excel_path, sheet_name="Batch_Fractures", index=False)
        print(f"Données exportées vers : {excel_path}")

    def analyze_folder(self, folder_path):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 28-07-2025 \n
        Description : Méthode simpliste d'analyse de joint \n
        :param folder_path: Chemin vers le fichier d'images à analyser
        :return: La position relative des runs et des fractures
        """
        all_core_runs = []

        image_files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.png")),
        key = last_4chars
    )

        if not image_files:
            print("Aucune image trouvée dans le dossier spécifié.")
            return

        # Parcourir et analyser chaque image
        for img_path in image_files:
            print(f"Analyse de l'image : {img_path}")
            core_runs_sorted = self.analyze_joints(img_path)

            # Ajouter chaque résultat clairement à une liste globale
            img_name = os.path.basename(img_path)
            for core_run in core_runs_sorted:
                core_run.img_name = img_name  # enregistrer le nom de l'image associée
                all_core_runs.append(core_run)

        # Export final vers Excel
        self.export_multiple_core_runs_to_excel(all_core_runs, "batch_analysis")

    def get_methods(self):
        """
        Accesseur des méthodes d'analyses, ajoutez une référence à votre méthode d'analyse ici afin de pouvoir l'utiliser dans l'interface.
        :return: Retourne un dictionnaire avec les méthodes d'analyses en clés
        """
        return {
            "Analyze core dataset (one img)": self.analyze_joints,
            "Analyze core dataset (multiple img)": self.analyze_folder,
            "Analyze and crop lab core sample": self.analyze_crop_sample,
            "Test avec image": self.analyze_example,
            # etc ...
        }


class CoreRun:
    """
    S'occupe des attributs d'analyse à l'intérieur d'un core (joint, broken core, lost core et faille)
    """
    def __init__(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.fractures = []  # une seule liste pour tous les types de fracture

        # autres attributs inchangés
        self.x_start_px = None
        self.x_end_px = None
        self.length_px = None

        self.start_m = None
        self.end_m = None
        self.length_m = None

    def add_fracture(self, fracture):
        self.fractures.append(fracture)


class Fracture:
    """
    Gère les attributs d'un joint (point milieu, point centre, boundary box, niveau de confiance).
    """
    def __init__(self, bbox, confidence, fracture_type):
        self.bbox = bbox
        self.confidence = confidence
        self.fracture_type = fracture_type  # 'joint', 'broken_core', 'lost_core', 'fault'

        self.distance_px = None
        self.distance_m = None
        self.length_px = abs(bbox[2] - bbox[0])  # longueur horizontale par défaut
        self.length_m = None

    def center(self):
        x_center = (self.bbox[0] + self.bbox[2]) / 2
        y_center = (self.bbox[1] + self.bbox[3]) / 2
        return x_center, y_center

    def center_x(self):
        return (self.bbox[0] + self.bbox[2]) / 2

    def compute_distances(self, core_bbox, pixel_to_meter_ratio):
        self.distance_px = self.center_x() - core_bbox[0]
        self.distance_m = self.distance_px * pixel_to_meter_ratio
        self.length_m = self.length_px * pixel_to_meter_ratio

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
    #return sorted(core_runs, key=lambda core: (core.bbox[0], core.bbox[1]))

    sorted_by_x = sorted(core_runs, key=lambda core: core.bbox[0])
    return sorted(sorted_by_x, key=lambda core: core.bbox[1]) # gauche à droite, haut en bas

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

def last_4chars(x):
    return x[-4:]

def clean_cache():
    """
    Vide la mémoire allouée à CUDA. Important d'appeller cette fonction au début de chaque méthode d'analyse.
    """
    torch.cuda.empty_cache()