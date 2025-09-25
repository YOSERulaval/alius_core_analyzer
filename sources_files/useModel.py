from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import os, cv2, glob, pandas as pd
from detectron2.utils.visualizer import Visualizer, ColorMode
from sympy.printing.pytorch import torch
from sources_files import excelExport as ee

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

        from sources_files import utilitiesMethods as utils
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
        Appelée via "analyze_example". # TODO conjugaison appeler lol, participe passé ?
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
        clean_cache() # Libérer la mémoire du GPU AVANT de commencer une analyse.
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # !! IMPORTANT de mettre votre 'note de passage' !!
        class_names = ["core", "joint", "broken_core", "lost_core", "fault"] # Ajoutez toutes les classes présentes sur vos annotations

        return self.__run_img_prediction(img_path, class_names)

    def analyze_cores(self, img_path, first_run=0.0):
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 06-08-2025 \n
        Description : Méthode appelée via "analyze_folder" qui analyse tous les cores dans une image ainsi que leurs
        fractures. \n
        :param first_run: Paramètre utilisé pour cumuler la profondeur du trou tout au long de l'analyse.
        :param img_path: Chemin vers l'image à analyser
        :return: La position relative des fractures, en ordre
        """
        clean_cache()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu") # instances enregistrées dans le buffer du cpu

        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        class_names = ["core", "joint", "broken_core", "lost_core", "fault"] # Les classes sont spécifiques à l'entraînement

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
                if bbox_contains_point(core.bbox, fracture.center()):
                    fracture.compute_distances(core.bbox, self.pixel_to_meter_ratio)
                    core.add_fracture(fracture)  # méthode unique add_fracture()

            cumulative_px += length_px
            core.real_core_measure()

            last_run = core.end_m

        return core_runs_sorted, float(last_run)

    def analyze_crop_sample(self, img_path):
        # TODO faire un training pour tester cette méthode
        """
        Auteur : Yohan Sercia \n
        Dernière MAJ : 06-08-2025 \n
        Méthode simpliste d'analyse de crop sample du lab \n
        ** Gracieuseté de ChatGPT, mais ça l'air legit **
        :param img_path: Chemin vers l'image à analyser
        """
        clean_cache()
        print("Analyzing core sample")
        #class_names = ["core"]
        #return self.__run_img_prediction(img_path, class_names)
        img = cv2.imread(img_path)
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu")

        if len(instances.pred_masks) == 0:
            print("Aucun masque détecté, cropping impossible.")
            return

        mask = instances.pred_masks[0].numpy().astype("uint8")
        y_coords, x_coords = mask.nonzero()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        cropped_img = img[y_min:y_max, x_min:x_max]

        output_filename = os.path.splitext(os.path.basename(img_path))[0] + "_cropped.png"
        cv2.imwrite(output_filename, cropped_img)
        print(f"Image cropée sauvegardée : {output_filename}")

        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def analyze_folder(self, folder_path):
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
        """
        Accesseur des méthodes d'analyses, ajoutez une référence à votre méthode d'analyse ici afin de pouvoir l'utiliser dans l'interface.
        :return: Retourne un dictionnaire avec les méthodes d'analyses en clés
        """
        return {
            "Analyze core dataset": self.analyze_folder,
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


# Fonctions globales, ne pas modifier TODO mettre dans un nouveau fichier à pars
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

def clean_cache():
    """
    Vide la mémoire allouée à CUDA. Important d'appeler cette fonction au début de chaque méthode d'analyse. \n
    Est une fonction stand alone au cas où on voudrait ajouter quelque chose d'autre.
    """
    torch.cuda.empty_cache()