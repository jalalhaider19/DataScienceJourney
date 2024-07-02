import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import numpy as np
import pandas as pd


dataset = foz.load_zoo_dataset("quickstart-geo")
session = fo.launch_app(dataset)

print(dataset.head())

##!pip install ultralytics - used a new software called ultralytics which is an extension of the project 

dataset = foz.load_zoo_dataset("quickstart-geo")
session = fo.launch_app(dataset)

print(dataset.head())


import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("quickstart-geo")
fob.compute_uniqueness(dataset)

# A list of ``[longitude, latitude]`` coordinates
locations = dataset.values("location.point.coordinates")

# Scalar `uniqueness` values for each sample
uniqueness = dataset.values("uniqueness")

# The number of ground truth objects in each sample
num_objects = dataset.values(F("ground_truth.detections").length())

# Create scatterplot
plot = fo.location_scatterplot(
    locations=locations,
    labels=uniqueness,      # color points by their `uniqueness` values
    sizes=num_objects,      # scale point sizes by number of objects
    labels_title="uniqueness",
    sizes_title="objects",
)
plot.show()

import os; os.environ["YOLO_VERBOSE"] = "False"

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.ultralytics as fou

from ultralytics import YOLO

# Load an example dataset
dataset = foz.load_zoo_dataset("quickstart-geo", max_samples=25)
dataset.select_fields().keep_fields()


import fiftyone as fo
import fiftyone.zoo as foz

MANHATTAN = [
    [
        [-73.949701, 40.834487],
        [-73.896611, 40.815076],
        [-73.998083, 40.696534],
        [-74.031751, 40.715273],
        [-73.949701, 40.834487],
    ]
]

dataset = foz.load_zoo_dataset("quickstart-geo")

#
# Create a view that only contains samples in Manhattan
#

view = dataset.geo_within(MANHATTAN)

# YOLOv8
model = YOLO("yolov8s.pt")
# model = YOLO("yolov8m.pt")
# model = YOLO("yolov8l.pt")
# model = YOLO("yolov8x.pt")

# YOLOv5
# model = YOLO("yolov5s.pt")
# model = YOLO("yolov5m.pt")
# model = YOLO("yolov5l.pt")
# model = YOLO("yolov5x.pt")

# YOLOv9
# model = YOLO("yolov9c.pt")
# model = YOLO("yolov9e.pt")

dataset.apply_model(model, label_field="boxes")

session = fo.launch_app(dataset)
for sample in dataset.iter_samples(progress=True):
    result = model(sample.filepath)[0]
    sample["car"] = fou.to_detections(result)
    sample.save()
    sample.display()

#pip install transformers extension for using Hugging Face
##Using Hugging Face & transformers 
import fiftyone as fo
import fiftyone.zoo as foz
dataset =  foz.load_zoo_dataset("quickstart-geo")

model = foz.load_zoo_model(
    "zero-shot-detection-transformer-torch",
    name_or_path="google/owlvit-base-patch32",  # HF model name or path
    classes=["plant", "tree", "car"]
)

dataset.apply_model(model, label_field="owl", confidence_thresh=0.05)

session = fo.launch_app(dataset, auto =False)

dataset.apply_model?
















