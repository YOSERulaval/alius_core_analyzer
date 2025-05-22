import trainModel as ac

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
ac.cfg.MODEL.WEIGHTS = ac.os.path.join(ac.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
ac.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = ac.DefaultPredictor(ac.cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = ac.load_coco_json(ac.val_ds_path, "datasets/val", "my_dataset_val")
for d in ac.random.sample(dataset_dicts, 3): # Tester le modèle avec 3 images
    im = ac.cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = ac.Visualizer(im[:, :, ::-1],
                   metadata=ac.core_val_metadata,
                   scale=0.3,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ac.cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
    ac.cv2.waitKey(0)