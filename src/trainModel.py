import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances, load_coco_json

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return DatasetEvaluators([COCOEvaluator(dataset_name, cfg, False, output_folder)])

class ModelTrainer:
    def __init__(self, num_iterations, num_classes, roihead_batch_size, real_batch_size,
                 learning_rate, model_device):
        self.cfg = get_cfg()
        point_rend.add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(r"detectron2\projects\PointRend\configs\InstanceSegmentation\pointrend_rcnn_R_101_FPN_3x_coco.yaml")
        #self.cfg.MODEL.WEIGHTS = r"C:\Users\yohan\PycharmProjects\alius_core_analyzer\machine_learning_model\datasets\model_final_ba17b9.pkl"

        # Training et evaluation
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ("my_dataset_val",)
        self.cfg.TEST.EVAL_PERIOD = 1000 # TODO faire varier

        self.cfg.DATALOADER.NUM_WORKERS = 2

        self.cfg.MODEL.DEVICE = model_device

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_classes

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roihead_batch_size

        #self.cfg.INPUT.MASK_FORMAT = "polygon" # ajouté pour tests 23-07-2025

        self.cfg.SOLVER.MAX_ITER = num_iterations
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
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
        # Configuration des paramètres d'évaluation avant de débuter l'entraînement
        split_coco_annotations(
            coco_json_path=coco_path,
            train_json_path="machine_learning_model/datasets/sample_cropper/json_files/coco_train.json", # Nom du nouveau fichier json qui sera créer pour l'entraînement
            val_json_path="machine_learning_model/datasets/sample_cropper/json_files/coco_val.json",  # Nom du nouveau fichier json qui sera créer pour l'évaluation
            split_ratio=0.8
        )

        # Assignations des méta-données à un dataset dynamiquement
        #register_coco_instances("my_dataset_train", {}, coco_path, dataset_path) # TODO mettre OO ready
        register_coco_instances("my_dataset_train", {}, "machine_learning_model/datasets/sample_cropper/json_files/coco_train.json", dataset_path)
        register_coco_instances("my_dataset_val", {}, "machine_learning_model/datasets/sample_cropper/json_files/coco_val.json", dataset_path)

        # On cherche (et trouve) le sous fichier "trained_models", sinon on le créé
        #root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = 'machine_learning_model'
        models_dir = os.path.join(root_dir, "trained_models")
        output_dir = os.path.join(models_dir, model_name)

        self.cfg.OUTPUT_DIR = output_dir # Assignation de ce path au modèle qui sera entrainé
        self.cfg.SEED = 1

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=False)

        #trainer = DefaultTrainer(self.cfg)
        trainer = MyTrainer(self.cfg)
        #DefaultTrainer.test(evaluators=)
        trainer.resume_or_load(resume=False)
        trainer.train()

def split_coco_annotations(coco_json_path, train_json_path, val_json_path, split_ratio=0.8, seed=42):
    import json
    import random
    import os
    import copy

    # pour que le split soit reproductible
    random.seed(seed)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    # Champs optionnels du format COCO officiel
    info = coco_data.get("info", {})
    licenses = coco_data.get("licenses", [])

    # Mélanger les images puis couper 80/20 (ou split_ratio)
    shuffled = images[:]  # copie
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * split_ratio)

    train_images = shuffled[:split_index]
    val_images = shuffled[split_index:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids]

    # Construire les fichiers JSON
    train_data = {
        "info": copy.deepcopy(info),
        "licenses": copy.deepcopy(licenses),
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    val_data = {
        "info": copy.deepcopy(info),
        "licenses": copy.deepcopy(licenses),
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }

    # Créer les dossiers si besoin
    os.makedirs(os.path.dirname(train_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_json_path), exist_ok=True)

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f)

    print(f"✅ Dataset divisé : {len(train_images)} images train, {len(val_images)} images val.")