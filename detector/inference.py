from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from config.config import Config


def get_keypoint_tuples(eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_detection_function(model):
    @tf.function
    def detect_fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def run_object_detection(image_path: str, config: Config):
    # Load pipeline and build a detection tool
    configurations = config_util.get_configs_from_pipeline_file(config.MODEL_CONFIG_PATH)
    model_configuration = configurations["model"]
    detection_model = model_builder.build(model_config=model_configuration, is_training=False)

    # Restore checkpoint
    checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint.restore(config.MODEL_CHECKPOINT_PATH).expect_partial()

    label_map = label_map_util.load_labelmap(config.LABEL_MAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map), use_display_name=True
    )

    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    image = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    detect_fn = get_model_detection_function(detection_model)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if "detection_keypoints" in detections:
        keypoints = detections["detection_keypoints"][0].numpy()
        keypoint_scores = detections["detection_keypoint_scores"][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"][0].numpy(),
        (detections["detection_classes"][0].numpy() + label_id_offset).astype(int),
        detections["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configurations["eval_config"]),
    )

    plt.figure(figsize=(12, 16))
    plt.imshow(image_np_with_detections)
    plt.savefig(image_path)
    # plt.show()
