# MURA Bone X-Ray Object Detection

Object detection on bone x-ray images from MURA dataset

## Setup

Before running any of the scripts export absolute path to the complete MURA dataset into `MURA_DATASET_ROOT_DIR`
environment variable. This path should point to parent directory of `original` and `generated` directories.

Tensorflow usage based on https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html

## Labeling images

Complete image material was taken from MURA dataset. Subset of those was needed for training the model, and in order to
mark regions of interest in those chosen images [LabelImg](https://github.com/tzutalin/labelImg) tool was used.

## Trained Model

From [Drive](https://drive.google.com/drive/folders/1V7sa2gpdz2vez1tpn4xVghrM7V-V8cjX?usp=sharing) download the
pretrained model and save it directly to `resources\object_detection\model`. Now you should have a directory like
`exported_ffrcnn_01` under the `model` directory.