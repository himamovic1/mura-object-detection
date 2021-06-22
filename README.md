# MURA Bone X-Ray Object Detection

Object detection on bone x-ray images from MURA dataset

## Setup

Before running any of the scripts export absolute path to the complete MURA dataset into `MURA_DATASET_ROOT_DIR`
environment variable. This path should point to parent directory of `original` and `generated` directories.

Model implementation based on [Tensorflow ObjectDetection API](https://github.com/tensorflow/models/tree/master/research/object_detection) as described in the [documentation/tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html).

Training and the development were done on the Windows 10 machine with an Nvidia GPU (enabled CUDA optimizations), running Python 3.9 and Tensorflow 2. Environment was set up according to the offical Tensorflow docs [Tensorflow installation](https://www.tensorflow.org/install) and [CUDA setup](https://www.tensorflow.org/install/gpu).

For the Neural Network model I opted for existing implementation of Faster RCNN already available at [Tensorflow Model ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Due to hardware restriction on my machine I chose the simplest Faster R-CNN model utilizing ResNet50 for 640x640 images .

## Trained Model

From [Drive](https://drive.google.com/drive/folders/1V7sa2gpdz2vez1tpn4xVghrM7V-V8cjX?usp=sharing) download the
pretrained model and save it directly to `resources\object_detection\model`. Now you should have a directory like
`exported_ffrcnn_01` under the `model` directory.

## Application

Custom trained model was mounted to a simple Flask web application which supports uploading the image, processing it by the model and showing the results both as a page content and on the image itself. This approach proved to be quite flexible since it's really simple to add new trained model checkpoints in the resources and redeploy the app to use the new model.

If I manage to find some time will deploy it to Heroku or something similar for easier access.

Page | Image
--- | --- 
Upload | ![Upload Form](https://github.com/himamovic1/mura-object-detection/blob/main/resources/screenshots/01.png)
--- | --- 
Result Negative | ![Result Negative Page](https://github.com/himamovic1/mura-object-detection/blob/main/resources/screenshots/03.png)
--- | --- 
Result Positive | ![Result Negative Page](https://github.com/himamovic1/mura-object-detection/blob/main/resources/screenshots/02.png)

## Labeling images

Complete image material was taken from MURA dataset. Subset of those was needed for training the model, and in order to
mark regions of interest in those chosen images [LabelImg](https://github.com/tzutalin/labelImg) tool was used.

Implant #1 | Implant #2
--- | --- 
![Label Implant](https://github.com/himamovic1/mura-object-detection/blob/main/resources/screenshots/04.png) | ![Label Implant Multiple](https://github.com/himamovic1/mura-object-detection/blob/main/resources/screenshots/05.png)



