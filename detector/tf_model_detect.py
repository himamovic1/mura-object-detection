from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from config.config import Config


def get_keypoint_tuples(eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_detection_function(model):
    @tf.function
    def detection_function_wrapper(img):
        """Preprocess and run detection on the given image."""
        img, shapes = model.preprocess(img)
        prediction_dict = model.predict(img, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detection_function_wrapper


def detect_and_mark_objects(image_path: str) -> List[Tuple[str, float]]:
    """
    Run object detection on a pretrained model.
    Mark any detected objects with a frame on the original image.
    Returns a list of found labels and matching scores (confidence).
    """
    # Load pipeline and build a detection tool
    configurations = config_util.get_configs_from_pipeline_file(Config.APP_CUSTOM_MODEL_TRAINED_CONFIG)
    model_configuration = configurations["model"]
    detection_model = model_builder.build(model_config=model_configuration, is_training=False)

    # Restore checkpoint
    checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint.restore(Config.APP_CUSTOM_MODEL_TRAINED_CHECKPOINT).expect_partial()

    label_map = label_map_util.load_labelmap(Config.LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map), use_display_name=True
    )

    category_index = label_map_util.create_category_index(categories)

    original_image = np.array(tf.keras.preprocessing.image.load_img(image_path))
    original_image_tensor = tf.convert_to_tensor(np.expand_dims(original_image, 0), dtype=tf.float32)

    detect_fn = get_model_detection_function(detection_model)
    detections, predictions_dict, shapes = detect_fn(original_image_tensor)

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if "detection_keypoints" in detections:
        keypoints = detections["detection_keypoints"][0].numpy()
        keypoint_scores = detections["detection_keypoint_scores"][0].numpy()

    # Prepare results
    boxes = detections["detection_boxes"][0].numpy()
    labels = (detections["detection_classes"][0].numpy() + Config.OBJECT_DETECTION_CLASSES_OFFSET).astype(int)
    scores = detections["detection_scores"][0].numpy()

    # Draw results on the original_image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        original_image,
        boxes,
        labels,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=Config.OBJECT_DETECTION_MIN_CONFIDENCE,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configurations["eval_config"]),
    )

    plt.figure(figsize=(12, 16))
    plt.imshow(original_image)
    plt.savefig(image_path)

    # Map labels and scores
    return [
        (Config.OBJECT_DETECTION_CLASSES[label], score)
        for label, score in zip(labels, scores)
        if score >= Config.OBJECT_DETECTION_MIN_CONFIDENCE
    ]
