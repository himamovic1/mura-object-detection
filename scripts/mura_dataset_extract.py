import os
import shutil
import time

from imutils import paths

from config.config import Config


def copy_positive_images(input_dir: str, output_dir: str, max_count: int) -> None:
    print(f"Extracting total of [{max_count}] images from [{input_dir}]")
    image_counter = 0
    for img_path in paths.list_images(input_dir):
        if "positive" in img_path:
            shutil.copy(img_path, os.path.join(output_dir, f"positive_{image_counter}.png"))
            image_counter += 1


if __name__ == "__main__":
    start = time.time()

    print("Ensure output paths exist")
    for directory in (Config.MURA_DATASET_OUTPUT_TRAIN_DIR, Config.MURA_DATASET_OUTPUT_TEST_DIR):
        if not os.path.exists(directory):
            os.makedirs(directory)

    print("Processing training set")
    copy_positive_images(
        input_dir=Config.MURA_DATASET_INPUT_TRAIN_DIR,
        output_dir=Config.MURA_DATASET_OUTPUT_TRAIN_DIR,
        max_count=Config.MURA_DATASET_TRAIN_IMAGE_COUNT,
    )

    print("Processing validation set")
    copy_positive_images(
        input_dir=Config.MURA_DATASET_INPUT_TEST_DIR,
        output_dir=Config.MURA_DATASET_OUTPUT_TEST_DIR,
        max_count=Config.MURA_DATASET_TEST_IMAGE_COUNT,
    )

    end = time.time()
    print(f"Processing completed in {(end - start):.2f} seconds")
