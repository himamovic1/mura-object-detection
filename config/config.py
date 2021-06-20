""" Main configuration file """
import logging
from os import path
from typing import Tuple, Dict

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
    DATASET_ROOT_PATH: str = path.join(mura_resources_path, "dataset")
    ANNOTATIONS_ROOT_PATH: str = path.join(mura_resources_path, "annotations")
    MODEL_ROOT_PATH: str = path.join(mura_resources_path, "model", "custom")
    MODEL_FRCNN_PATH: str = path.join(mura_resources_path, "model", "exported_frcnn_01")

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
    MODEL_CONFIG_PATH: str = path.join(MODEL_ROOT_PATH, "pipeline.config")
    MODEL_CHECKPOINT_PATH: str = path.join(MODEL_FRCNN_PATH, "checkpoint", "ckpt-0")

    # Detection process config
    OBJECT_DETECTION_MIN_CONFIDENCE: float = 0.85
    OBJECT_DETECTION_CLASSES_OFFSET: int = 1
    OBJECT_DETECTION_CLASSES: Dict[int, str] = {
        1: "fracture",
        2: "implant"
    }
