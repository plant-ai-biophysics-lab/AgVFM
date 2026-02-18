# %%
# TODO: Modify to allow testing on all images in experiment sets to get overall performance metrics for zero-shot case
# Next: add benchmarking of Grounding-Dino, OWL-V2
# After: add finetuning on k synthetic train examples; test again (randomly select the k examples)
# synthetic_path = "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/RawData/Syn"
# syn labels in coco format are under synthetic_path, images in images dir
# Also need to test performance on pods; 
# maybe also performance with both pod and flower in query (need to update logic for results if we do) 

# --> add imports to conda env as needed


## **COMMENTED OUT PREPPING YOLO LABELS; UNCOMMENT FOR POD DATA LATER**


# import os
# labels_path="/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_location_year_genotype/test/labels"
# # image_path="/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_2022_location_Unseen_2023_genotype/test/images"

# # %%
# from ultralytics.data.converter import convert_coco
# import os

# image_paths.append(os.path.abspath(os.path.join(labels_path, "../images")))
# # labels_path is a file path, so we take dirname first to get the folder
# base_dir = os.path.dirname(labels_path)
# # create save_dir as a sibling to 'labels' (path/to/test/yolo_labels)
# save_dir = os.path.abspath(os.path.join(base_dir, "../yolo_labels"))

# convert_coco(
#     labels_dir=labels_path,
#     save_dir=save_dir,
#     use_keypoints=False,
# )
# # Update labels_path to point to the directory where yolo labels are saved
# # convert_coco creates 'labels' subdirectory inside save_dir and saves the yolo annotations inside a {coco_filename} subdir
# # where {coco_filename} is the name of the coco file without .json

# labels_path = os.path.join(save_dir, "labels/_annotations.coco")

# label_conversion_paths = ["/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_2023_Location_Genotype_Unseen_2022/test/labels", 
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_2022_location_Unseen_2023_genotype/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_2023_Location_Genotype_Unseen_2022/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_2023_location_Unseen_2022_genotype/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_davis_year_genotype_Unseen_kearney/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_davis_year_Unseen_kearney_genotype/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_kearney_year_genotype_Unseen_davis/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_kearney_year_Unseen_davis_genotype/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_location_year_genotype/test/labels",
# "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_location_year_Unseen_genotype/V0/test/labels"
# ]

# yolo_label_paths = []
# image_paths = []
# # in all label_paths go to ../images to get images

# for labels_path in label_conversion_paths:
#     image_paths.append(os.path.abspath(os.path.join(labels_path, "../images")))
#     # labels_path is a file path, so we take dirname first to get the folder
#     base_dir = os.path.dirname(labels_path)
#     # create save_dir as a sibling to 'labels' (path/to/test/yolo_labels)
#     save_dir = os.path.abspath(os.path.join(base_dir, "../yolo_labels"))
    
#     convert_coco(
#         labels_dir=labels_path,
#         save_dir=save_dir,
#         use_keypoints=False,
#     )
#     # Update labels_path to point to the directory where yolo labels are saved
#     # convert_coco creates 'labels' subdirectory inside save_dir and saves the yolo annotations inside a {coco_filename} subdir
#     # where {coco_filename} is the name of the coco file without .json

#     labels_path = os.path.join(save_dir, "labels/_annotations.coco")
#     yolo_label_paths.append(labels_path)

# %%



## **COMMENTED OUT EVAL W/ PLOTTING / LOOPING LOGIC



# from pathlib import Path

# from vlme import run_gt_vs_predictions

# image_path = image_paths[i]
# labels_path = yolo_label_paths[i]
# print(labels_path)
# # --- User inputs ---
# NUM_IMAGES = 2
# RANDOM_SEED = 101
# IMAGES_DIR = image_path
# LABELS_DIR = labels_path
# MODEL_WEIGHTS_PATH = "model_weights/yolov8x-worldv2.pt" 
# GT_CLASS_NAMES = ["flower"]
# OPEN_SET_CLASS_NAMES = [
#     "small yellow flower", 
#     "small white flower",
#     ""
# ]
# PREDICT_KWARGS = {
#     "imgsz": 1280,
#     "conf": 0.1,
#     "max_det": 500,
#     "verbose": False,
#     "iou": 0.3
# }

# _, _, results_list = run_gt_vs_predictions(
#     images_dir=IMAGES_DIR,
#     labels_dir=LABELS_DIR,
#     num_images=NUM_IMAGES,
#     random_seed=RANDOM_SEED,
#     class_names=GT_CLASS_NAMES,
#     yolo_classes=OPEN_SET_CLASS_NAMES,
#     weights_path=MODEL_WEIGHTS_PATH,
#     predict_kwargs=PREDICT_KWARGS,
#     ncols=2,
#     show=True,
#     return_results=True,
# )

# for i in range(len(image_paths)):
    # image_path = image_paths[i]
    # labels_path = yolo_label_paths[i]
    # print(labels_path)
    # # --- User inputs ---
    # NUM_IMAGES = 2
    # RANDOM_SEED = 101
    # IMAGES_DIR = image_path
    # LABELS_DIR = labels_path
    # MODEL_WEIGHTS_PATH = "model_weights/yolov8x-worldv2.pt" 
    # GT_CLASS_NAMES = ["flower"]
    # OPEN_SET_CLASS_NAMES = [
    #     "small yellow flower", 
    #     "small white flower",
    #     ""
    # ]
    # PREDICT_KWARGS = {
    #     "imgsz": 1280,
    #     "conf": 0.1,
    #     "max_det": 500,
    #     "verbose": False,
    #     "iou": 0.3
    # }

    # _, _, results_list = run_gt_vs_predictions(
    #     images_dir=IMAGES_DIR,
    #     labels_dir=LABELS_DIR,
    #     num_images=NUM_IMAGES,
    #     random_seed=RANDOM_SEED,
    #     class_names=GT_CLASS_NAMES,
    #     yolo_classes=OPEN_SET_CLASS_NAMES,
    #     weights_path=MODEL_WEIGHTS_PATH,
    #     predict_kwargs=PREDICT_KWARGS,
    #     ncols=2,
    #     show=True,
    #     return_results=True,
    # )

# %%
from vlme import run_evaluation
from pathlib import Path

# NUM_IMAGES_TO_EVAL = 10
# NUM_IMAGES_TO_PLOT = 10
RANDOM_SEED = 100
#IMAGES_DIR = image_path
IMAGES_DIR = "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_location_year_genotype/test/images"
# LABELS_DIR = labels_path
LABELS_DIR = "/group/jmearlesgrp/GEMINI/lars/grounding/AgVFM/_data/Flower/Exps/Seen_location_year_genotype/yolo_labels/labels/_annotations.coco"
# MODEL_WEIGHTS_PATH = "../model_weights/yolov8x-worldv2.pt"
GT_CLASS_NAMES = ["flower"]

# need to separate classes with . and end with . for GD; handled in hugging_face.py
# normal list for OWL
OPEN_SET_CLASS_NAMES = [
    "small yellow flower", 
    "small white flower"
]
PREDICT_KWARGS = {
    "imgsz": 1280,
    "conf": 0.1,
    "max_det": 500,
    "verbose": False,
    "iou": 0.5
}


# removed NUM_IMAGES_TO_EVAL to run on all images in the set for overall performance metrics for zero-shot case
model_list = ["yolo", "grounding_dino", "owlv2"]
weights_dict = {"yolo": "../model_weights/yolov8x-worldv2.pt",
                "grounding_dino": "IDEA-Research/grounding-dino-base",
                "owlv2": "google/owlvit-large-patch14"
}
for MODEL in model_list:
    if MODEL == "yolo":
        CURR_CLASS_NAMES = OPEN_SET_CLASS_NAMES + [""] # bg class
    else:
        CURR_CLASS_NAMES = OPEN_SET_CLASS_NAMES # no bg class for GD/OWL
    metrics = run_evaluation(
        images_dir=IMAGES_DIR, 
        labels_dir=LABELS_DIR,
        random_seed=RANDOM_SEED,
        class_names=GT_CLASS_NAMES,
        yolo_classes=CURR_CLASS_NAMES,
        weights_path=weights_dict[MODEL],
        model_type=MODEL,
        plot_summary=False,
        plot_pr_curve=False, # no plotting in this case
        plot_sample=0, # ^^
        predict_kwargs=PREDICT_KWARGS
    )

    print(f"Evaluation metrics from {MODEL}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, mAP@{PREDICT_KWARGS['iou']}: {metrics['map']:.4f}")

# Modified args for run_evaluation:
# Args:
#         images_dir: Directory of test images.
#         labels_dir: Directory of YOLO .txt labels (same stem as images).
#         num_images: If None, use full set of images; else use this many (sampled with random_seed).
#         random_seed: Seed for sampling when num_images is not None.
#         class_names: Names for GT class IDs (default ["flower"]).
#         yolo_classes: Text prompts for YOLO World (default OPEN_SET_CLASS_NAMES).
#         **NEW**: weights_path: Path to YOLO World .pt weights or HF model ID. (default still is yolov8x-worldv2.pt)
#         **NEW**: model_type: "yolo", "grounding_dino", or "owlv2".
#         predict_kwargs: Optional dict for model.predict() (merged over defaults).
#         iou_threshold: IoU threshold for mAP/P/R (default 0.5).
#         plot_summary: If True, show metrics summary figure.
#         plot_pr_curve: If True and plot_summary, include P-R curve in summary.
#         plot_sample: If > 0, also run plot_gt_vs_predictions on this many sampled images (separate figure).
#         show: Whether to call plt.show() on figures.


# Returned metrics from run_evaluation:

# out = dict(metrics)
# out["image_paths"] = image_paths
# out["list_gt_xyxy"] = list_gt_xyxy
# out["list_pred_xyxy"] = list_pred_xyxy
# out["list_pred_conf"] = list_pred_conf
# return out

