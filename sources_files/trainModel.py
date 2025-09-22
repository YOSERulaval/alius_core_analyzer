import os, json, cv2, random
import tensorboard
import torch, detectron2

# import some common libraries
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from detectron2.data.datasets import register_coco_instances, load_coco_json

class ModelTrainer:
    def __init__(self, num_iterations, num_classes, roihead_batch_size, real_batch_size,
                 learning_rate, model_device):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ()

        self.cfg.DATALOADER.NUM_WORKERS = 4

        self.cfg.MODEL.DEVICE = model_device
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roihead_batch_size
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        #self.cfg.INPUT.MASK_FORMAT = "polygon" # ajouté pour tests 23-07-2025

        self.cfg.SOLVER.MAX_ITER = num_iterations
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        self.cfg.SOLVER.STEPS = []
        #self.cfg.SOLVER.REFERENCE_WORLD_SIZE = 1 # number of workers (GPUs) this config is meant to train with. ! n'est pas implémenté encore !
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MOMENTUM = 0.9
        self.cfg.SOLVER.IMS_PER_BATCH = real_batch_size

        # Légende des paramètres :
        """
        ===============================================================================
        Légende des paramètres, tiré du tutoriel de detectron2, disponible sur collab :
        (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
        ===============================================================================
        cfg.MODEL.WEIGHTS: Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH: Number of images per batch across all machines. This is also the number of training images per step (i.e. per iteration).
        cfg.SOLVER.BASE_LR: pick a good LR
        cfg.SOLVER.MAX_ITER: 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []: do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE: The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES: only has one class (ballon)
        """

    def begin_training(self, coco_path, dataset_path, model_name):
        # Assignations des méta-données à un dataset dynamiquement
        register_coco_instances("my_dataset_train", {}, coco_path, dataset_path)

        # On cherche (et trouve) le sous fichier "trained_models", sinon on le créé
        #root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = 'machine_learning_model'
        models_dir = os.path.join(root_dir, "trained_models")
        output_dir = os.path.join(models_dir, model_name)

        self.cfg.OUTPUT_DIR = output_dir # Assignation de ce path au modèle qui sera entrainé
        self.cfg.SEED = 1

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=False)

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

"""
# Paths des datasets
def train_path_picker():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(title="Choisir le chemin du fichier json d'entrainement (train)",
                                      filetypes=(("json files", ".json"),))
    root.destroy()

    if not path:
        print("Aucun fichier sélectionné, sortie.")
        exit(0)

    return path
train_ds_path = train_path_picker()

# Assignations des méta-données à un dataset
register_coco_instances("my_dataset_train", {}, train_ds_path, "datasets/train")

# Getter pour les métadonnées
core_train_metadata = MetadataCatalog.get("my_dataset_train")

# Affiche les images avec les métadonnées (pour tests)
dataset_dicts = load_coco_json(train_ds_path, "datasets/train", "my_dataset_train")

def show_metadata(ds_dicts):
    for d in random.sample(ds_dicts, 3): # Vérifier les annotations pour 3 images
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=core_train_metadata, scale=0.25)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

# !! Réviser chacun des paramètres pour connaître leur incidence sur le modèle !!
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people, 2 by default
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()

def main():
    while True:
        print(
            "\n"
            "=============================================================\n"
            "====================== WILLY TRAINER 2006 ===================\n"
            "=============================================================\n"
            "I : Importer les datasets 'train' et 'val'\n"
            "-------------------------------------------------------------\n"
            "T : Commencer l'entraînement  |  S : Afficher les métadonnées\n"
            "Q : Quitter\n"
            "=============================================================\n"
            "\n"
        )
        choix = input("Votre choix (I/T/S/Q) : ").strip().upper()
        if choix == "I":
            print("En construction !")

        elif choix == 'T':
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
        elif choix == 'S':
            show_metadata(dataset_dicts)
        elif choix == 'Q':
            print("Bye bye !")
            break
        else:
            print("Mauvais choix\n")


if __name__ == '__main__':
    main()

"""