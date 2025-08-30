# import de trainModel.py
from sources_files import trainModel as tm

# import some common libraries

# import some common detectron2 utilities
from detectron2.utils.visualizer import ColorMode

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
tm.cfg.MODEL.WEIGHTS = tm.os.path.join(tm.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
tm.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = tm.DefaultPredictor(tm.cfg)

# Paths des datasets
def val_path_picker():
    root = tm.tk.Tk()
    root.withdraw()

    path = tm.filedialog.askopenfilename(title="Choisir le chemin du fichier json de test (val)",
                                          filetypes=(("json files", ".json"),))
    root.destroy()

    if not path:
        print("Aucun fichier sélectionné, sortie.")
        exit(0)

    return path
val_ds_path = val_path_picker()

# Assignations des méta-données à un dataset
tm.register_coco_instances("my_dataset_val", {}, val_ds_path, "datasets/val")

# Getter pour les métadonnées
core_val_metadata = tm.MetadataCatalog.get("my_dataset_val")

# Affiche les images avec les métadonnées
dataset_dicts = tm.load_coco_json(val_ds_path, "datasets/val", "my_dataset_val")

for d in tm.random.sample(dataset_dicts, 3): # Tester le modèle avec 3 images
    im = tm.cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    v = tm.Visualizer(im[:, :, ::-1],
                       metadata=core_val_metadata,
                       scale=0.2,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    tm.cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
    tm.cv2.waitKey(0)