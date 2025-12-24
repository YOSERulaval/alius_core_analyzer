"""
Alius Consultation Minière inc.
====================================
CRÉÉ PAR:       Yohan Sercia
CRÉE LE:        JJ-MM-2025
------------------------------------
PROJET:         Geotech IA
DESCRIPTION:    [DESCRIPTION]
------------------------------------
DERNIÈRE MAJ:   17-12-2025
====================================
"""
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2, glob, pandas as pd
from detectron2.utils.visualizer import Visualizer, ColorMode
from numpy.ma.core import append
from sympy.printing.pytorch import torch

# Fichiers à importer de 'sources_files'
from sources_files import excelExport as ee
from sources_files import methodLinker as mlinker
from sources_files import utilitiesMethods as utils

# TODO revoir l'organisation des fonctions/méthodes
class ModelPredictor:
    def __init__(self, model_dir,  num_classes, confidence_thresh=0.8, output_path='', project_name='Project COG-25-12'): # TODO mettre dynamique
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

        # Nommer et créer le dossier des outputs du projet
        self.output_path = output_path
        self.project_name = project_name

        self.output_path = utils.make_new_dir(project_name)

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

    def __run_img_prediction(self, img_path, class_names):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 06-08-2025 \n
        Description : Méthode privée, permet d'afficher l'image avec les masks de prédictions. Utile pour tests et démo. \n
        Appelée via "analyze_example".
        :param img_path: Chemin vers l'image à analyser
        :return: class_names, class_area, img_name
        """
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        outputs = self.predictor(img)

        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu
        prediction_masks = instances.pred_masks.cpu().numpy()
        detected_class_indexes = instances.pred_classes.cpu().numpy()

        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        class_catalog = metadata.thing_classes

        for idx, coordinates in enumerate(prediction_masks):
            class_index = detected_class_indexes[idx]
            class_name = class_catalog[class_index]
            class_names.append(class_name)

        print(class_names)

        #masks = instances.pred_masks.cpu().numpy()
        #classes = instances.pred_classes.cpu().numpy()

        #class_area = {class_name: 0 for class_name in class_names}

        #for mask, class_id in zip(masks, classes):
        #    class_name = class_names[class_id]
        #    area_pixels = np.sum(mask)
        #    class_area[class_name] += area_pixels

        # Visualisation du résultat
        v = Visualizer(img[:, :, ::-1], scale=0.3, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances.to("cpu")) # image dans buffer cpu, peut être qu'on peut générer dans GPU ?
        cv2.imshow(img_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __output_prediction(self, img_path, output_dir, class_names=None):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 22-09-2025 \n
        Description : Méthode privée, permet de sauvegarder l'image avec les masks de prédictions. \n
        :param img_path: Chemin vers l'image à analyser
        :param output_dir:
        :param class_names: Nom de la ou les classes présentes dans l'annotation
        """
        from detectron2.data import MetadataCatalog

        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu")

        # Préparer la metadata pour afficher des noms de classes lisibles
        metadata = None
        if class_names:
            # On utilise un catalogue temporaire dédié à l'inférence
            MetadataCatalog.get("__inference__").set(thing_classes=class_names)
            metadata = MetadataCatalog.get("__inference__")

        # Dessin de l'overlay (RGB pour Visualizer, BGR pour OpenCV)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(instances)
        overlay = out.get_image()[:, :, ::-1]  # repasser en BGR pour cv2.imwrite

        # Nom de sortie : <nom>_pred.<ext>
        stem, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(output_dir, f"{stem}_pred{ext or '.png'}")
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(out_path, overlay)
        print(f"Overlay sauvegardé : {out_path}")

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
        class_names = ["CORE", "JN", "BC", "FLT", "TB", "DK"]
        utils.clean_cache() # Libérer la mémoire du GPU AVANT de commencer une analyse.
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # !! IMPORTANT de mettre votre 'note de passage' !!

        return self.__run_img_prediction(img_path, class_names)

    def analyze_cores(self, img_path, first_run=0.0):
        # TODO migrer vers APT/APT_CoreLogging_YS.py
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 06-08-2025 \n
        Description : Méthode appelée via "analyze_folder" qui analyse tous les cores dans une image ainsi que leurs
        fractures. \n
        :param first_run: Paramètre utilisé pour cumuler la profondeur du trou tout au long de l'analyse.
        :param img_path: Chemin vers l'image à analyser
        :return: La position relative des fractures, en ordre
        """
        utils.clean_cache()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu

        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        class_names = ["core", "joint", "broken_core", "fault", "table", "disking"] # Les classes sont spécifiques à l'entraînement

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
        core_runs_sorted = utils.sort_cores(core_runs) # Tri des cores de haut en bas et gauche à droite

        cumulative_px = 0 # On commence l'analyse à zéro. La vraie profondeur est ajouté via Excel après

        self.pixel_to_meter_ratio = utils.pixel_to_meter(core_runs_sorted, 0.0635) # TODO pouvoir donner le choix entre HQ, NQ, HQ3 dans l'interface

        last_run = first_run

        # Boucle principale qui calcul les mesures cumulées
        for core in core_runs_sorted:
            x1, _, x2, _ = core.bbox
            length_px = x2 - x1

            core.x_start_px = cumulative_px
            core.x_end_px = cumulative_px + length_px
            core.length_px = length_px

            core.start_m = first_run + (cumulative_px * self.pixel_to_meter_ratio)
            core.end_m = first_run + (core.x_end_px * self.pixel_to_meter_ratio)
            core.length_m = length_px * self.pixel_to_meter_ratio

            # Associer chaque fracture à son core
            for fracture in fractures:
                if utils.bbox_contains_point(core.bbox, fracture.center()):
                    fracture.compute_distances(core.bbox, self.pixel_to_meter_ratio)
                    core.add_fracture(fracture)  # méthode unique add_fracture()

            cumulative_px += length_px
            core.real_core_measure()

            last_run = core.end_m

        return core_runs_sorted, float(last_run)

    def analyze_folder(self, folder_path):
        # TODO migrer vers APT/APT_CoreLogging_YS.py
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 06-08-2025 \n
        Description : Appelle "analyze_cores" pour chaque image se trouvant à l'adresse de folder_path. \n
        Appelle ensuite "export_core_runs_to_excel" pour exporter les données dans un fichier Excel.
        :param folder_path: Chemin vers le fichier dans lequel se trouve les images à analyser
        """
        all_core_runs = []

        image_files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.png")),
        key = lambda name: name[-8:] # _001.jpg, _002.jpg, _003.jpg, etc...
)                                    # important sinon '101.05-112.03 (D)' passe avant '46.00-58.40 (D)'

        if not image_files:
            print("Aucune image trouvée dans le dossier spécifié.")
            return

        # Parcourir et analyser chaque image
        i = 0
        first_run = 0.0
        very_first_run = 40.97 # TODO mettre dynamique
        for img_path in image_files:
            print(f"Analyse de l'image : {img_path}")
            if i >= 1:
                core_runs_sorted, last_run = self.analyze_cores(img_path, first_run)
                self.__output_prediction(img_path, self.output_path)
            else:
                core_runs_sorted, last_run = self.analyze_cores(img_path, very_first_run)
                self.__output_prediction(img_path, self.output_path)
            first_run = last_run
            i += 1


            # Ajouter chaque résultat à une liste globale
            img_name = os.path.basename(img_path)
            for core_run in core_runs_sorted:
                core_run.img_name = img_name  # enregistrer le nom de l'image associée
                all_core_runs.append(core_run)

        # Export final vers Excel
        ee.export_core_runs_to_excel(all_core_runs, "batch_analysis", self.output_path)

    def get_methods(self):
        # Méthode built in:
        builtins = {
            "Analyze core dataset": self.analyze_folder, #TODO à implémenter dans APT
            "Test avec image": self.analyze_example,
        }

        # Ajoute automatiquement tous les APT du dossier /APT
        plugins = mlinker.build_apt_callables(model=self)

        # création d'un dictionnaire avec, d'abord, les méthodes built in et ensuite les APT ajoutés
        return {**builtins, **plugins}

class CoreRun:
    # TODO migrer vers APT/APT_CoreLogging_YS.py
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

        self.fractures_length = None
        self.total_rqd_length = None
        self.core_rqd = None

    def add_fracture(self, fracture):
        self.fractures.append(fracture)

    def real_core_measure(self):
        fracture_positions = sorted([f.distance_m for f in self.fractures]) # on tri d'abord toutes les fractures dans la liste
        fracture_positions = [0] + fracture_positions + [self.length_m] # Ajout de bornes inférieur-extérieur

        self.total_rqd_length = self.length_m # On commence avec la longueur totale du core

        for idx in range(len(fracture_positions) - 1):
            segment_length = fracture_positions[idx + 1] - fracture_positions[idx] # calcul de la diff entre deux joints

            # Calcul du RQD: On retire tous les éléments de moins de 10 cm
            if segment_length < 0.1:
                self.total_rqd_length -= segment_length

        for fracture in self.fractures:
            if fracture.fracture_type != "joint":
                self.total_rqd_length -= fracture.length_m

        self.core_rqd = (self.total_rqd_length / self.length_m) * 100 # Calcul du RQD


class Fracture:
    # TODO migrer vers APT/APT_CoreLogging_YS.py
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