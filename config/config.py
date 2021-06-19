""" Main configuration file """
import logging
from os import path
from typing import Tuple, Dict

project_root = path.dirname(path.dirname(path.abspath(__file__)))
resources_root = path.join(project_root, "resources")


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
    MURA_DATASET_PATH: str = "/Users/haris.imamovic/Desktop/Master/Bone/Dataset"
    MURA_DATASET_TRAIN_CSV_REGISTRY: str = path.join(MURA_DATASET_PATH, "MURA-v1.1", "train_image_paths.csv")
    MURA_DATASET_VALID_CSV_REGISTRY: str = path.join(MURA_DATASET_PATH, "MURA-v1.1", "valid_image_paths.csv")
    MURA_DATASET_IMAGE_SIZE: Tuple[int, int] = (320, 320)
    MURA_DATASET_LABELS: Dict[int, str] = {0: "negative", 1: "positive"}

    # Selective search config
    MAX_REGION_PROPOSALS_TRAIN: int = 2000
    MAX_REGION_PROPOSALS_INFER: int = 200

    # Network model configuration
    MODEL_PATH: str = path.join(resources_root, "model", "model.h5")
    ENCODER_PATH: str = path.join(resources_root, "model", "label_encoder.pickle")
    SCORING_PLOT_PATH: str = path.join(resources_root, "model", "plot.png")

    INPUT_IMAGE_DIMENSIONS: Tuple[int, int] = (224, 224)
    INITIAL_LEARNING_RATE: float = 1e-4
    NUMBER_OF_EPOCHS: int = 5
    BATCH_SIZE: int = 32
    MIN_CONFIDENCE: float = 0.99
