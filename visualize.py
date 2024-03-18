#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from detectron2.data.datasets import register_coco_instances


register_coco_instances("prima_val",{},"output.json","/data2/users/abanerjee/prima/val")
input = "coco_instances_results_PRIMA_MNv2_R50.json"
output = "mnv2_r50"
conf_threshold = 0.5

with PathManager.open(input, "r") as f:
  predictions = json.load(f)

pred_by_image = defaultdict(list)
for p in predictions:
  pred_by_image[p["image_id"]].append(p)

dicts = list(DatasetCatalog.get("prima_val"))
metadata = MetadataCatalog.get("prima_val")
if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
  def dataset_id_map(ds_id):
    return metadata.thing_dataset_id_to_contiguous_id[ds_id]

os.makedirs(output, exist_ok=True)

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

for dic in tqdm.tqdm(dicts):
  img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
  basename = os.path.basename(dic["file_name"])
  predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
  vis = Visualizer(img, metadata)
  vis_pred = vis.draw_instance_predictions(predictions).get_image()
  vis = Visualizer(img, metadata)
  vis_gt = vis.draw_dataset_dict(dic).get_image()
  concat = np.concatenate((vis_pred, vis_gt), axis=1)
  cv2.imwrite(os.path.join(output, basename), concat[:, :, ::-1])