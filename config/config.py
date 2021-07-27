""" Main configuration file """
import logging
from os import path, environ
from typing import Dict

project_root = path.dirname(path.dirname(path.abspath(__file__)))
resources_root = path.join(project_root, "resources")
mura_resources_path = path.join(resources_root, "object_detection")


class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = "SUPER_SECRET_KEY"
    SESSION_TYPE = "memcached"

    # Flask server relevant locations
    FLASK_TEMPLATES_PATH: str = path.join(resources_root, "templates")
    FLASK_STATIC_PATH: str = path.join(resources_root, "static")

    # Logging
    LOGGING_LEVEL: int = logging.DEBUG
    LOGGING_FILE: str = path.join(resources_root, "log_archive.log")

    # Dataset
    DATASET_ROOT_PATH: str = path.join(mura_resources_path, "dataset", "implants_fractures")
    ANNOTATIONS_ROOT_PATH: str = path.join(mura_resources_path, "annotations", "implants_fractures")
    CUSTOM_MODEL_ROOT_PATH: str = path.join(mura_resources_path, "model", "custom_faster_rcnn_implants_fractures")
    MODEL_FRCNN_PATH: str = path.join(mura_resources_path, "model", "tensorflow_faster_rcnn_resnet50")

    # Derived paths
    DATASET_PATH: Dict[str, str] = {
        "train": path.join(DATASET_ROOT_PATH, "train"),
        "test": path.join(DATASET_ROOT_PATH, "test"),
    }

    TF_RECORDS_PATH: Dict[str, str] = {
        "train": path.join(ANNOTATIONS_ROOT_PATH, "train.record"),
        "test": path.join(ANNOTATIONS_ROOT_PATH, "test.record"),
    }

    LABEL_MAP_PATH: str = path.join(ANNOTATIONS_ROOT_PATH, "label_map.pbtxt")
    MODEL_CONFIG_PATH: str = path.join(CUSTOM_MODEL_ROOT_PATH, "pipeline.config")
    MODEL_CHECKPOINT_PATH: str = path.join(MODEL_FRCNN_PATH, "checkpoint", "ckpt-0")

    # Detection process config
    OBJECT_DETECTION_MIN_CONFIDENCE: float = 0.85
    OBJECT_DETECTION_CLASSES_OFFSET: int = 1
    OBJECT_DETECTION_CLASSES: Dict[int, str] = {1: "implant", 2: "fracture"}

    ########################################################
    # Tensorflow ObjectDetection API Training & Evaluating #
    ########################################################
    TF_FLAG_MODEL_DIR: str = CUSTOM_MODEL_ROOT_PATH
    TF_FLAG_CHECKPOINT_DIR: str = CUSTOM_MODEL_ROOT_PATH
    TF_FLAG_PIPELINE_CONFIG_PATH: str = MODEL_CONFIG_PATH
    TF_FLAG_NUM_TRAINING_STEPS: int = None
    TF_FLAG_EVAL_INTERVAL: int = 300
    TF_FLAG_EVAL_TIMEOUT: int = 3600
    TF_FLAG_EVAL_ON_TRAIN_DATA: bool = False
    TF_FLAG_SAMPLE_1_OF_N_EVAL_EXAMPLES: int = None
    TF_FLAG_SAMPLE_1_OF_N_EVAL_ON_TRAIN_EXAMPLES: int = 5
    TF_FLAG_USE_TPU: bool = False
    TF_FLAG_TPU_NAME: str = None
    TF_FLAG_NUM_WORKERS: int = 1
    TF_FLAG_CHECKPOINT_EVERY_N: int = 1000
    TF_FLAG_RECORD_SUMMARIES: bool = True

    ########################################################
    # Tensorflow ObjectDetection API Model Exporting       #
    ########################################################
    TF_FLAG_IMAGE_INPUT_TYPE: str = "image_tensor"
    TF_FLAG_USE_SIDE_INPUTS: bool = False
    TF_FLAG_SIDE_INPUT_NAMES: str = ""
    TF_FLAG_SIDE_INPUT_TYPES: str = ""
    TF_FLAG_SIDE_INPUT_SHAPES: str = ""

    # TODO: Always update the output path before exporting to avoid overriding models
    TF_FLAG_CUSTOM_MODEL_TRAINED_OUTPUT_DIR: str = path.join(
        mura_resources_path, "model", "exported_custom_model_02_implants_fractures")

    ########################################################
    # Main Application Detector Configuration              #
    ########################################################
    APP_CUSTOM_MODEL_TRAINED_DIR: str = TF_FLAG_CUSTOM_MODEL_TRAINED_OUTPUT_DIR
    APP_CUSTOM_MODEL_TRAINED_CONFIG: str = path.join(APP_CUSTOM_MODEL_TRAINED_DIR, "pipeline.config")
    APP_CUSTOM_MODEL_TRAINED_CHECKPOINT: str = path.join(APP_CUSTOM_MODEL_TRAINED_DIR, "checkpoint", "ckpt-0")

    ########################################################
    # MURA Dataset (to extract test and validation data)   #
    ########################################################
    MURA_DATASET_ROOT_DIR: str = environ.get("MURA_DATASET_ROOT_DIR")
    MURA_DATASET_INPUT_TRAIN_DIR: str = path.join(MURA_DATASET_ROOT_DIR, "original", "train", "XR_HUMERUS")
    MURA_DATASET_INPUT_TEST_DIR: str = path.join(MURA_DATASET_ROOT_DIR, "original", "valid", "XR_HUMERUS")
    MURA_DATASET_OUTPUT_TRAIN_DIR: str = path.join(MURA_DATASET_ROOT_DIR, "generated", "train")
    MURA_DATASET_OUTPUT_TEST_DIR: str = path.join(MURA_DATASET_ROOT_DIR, "generated", "test")

    MURA_DATASET_TRAIN_IMAGE_COUNT: int = 500
    MURA_DATASET_TEST_IMAGE_COUNT: int = 150
