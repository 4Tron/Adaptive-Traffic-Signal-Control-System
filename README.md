# Adaptive Traffic Signal Control System

## YoloV3 Car counter
This is a demo project that uses YoloV3 neural network to count vehicles on a given video. The detection happens every x frames where x can be specified. Other times the dlib library is used for tracking previously detected vehicles. Furthermore, you can edit confidence detection level, number of frames to count vehicle as detected before removing it from trackable list and the maximum distance from centroid (see CentroidTracker class), number of frames to skip detection (and only use tracking) and the whether to use the original video size as annotations output or the YoloV3 416x416 size.

YoloV3 model is pretrained and downloaded (Internet connection is required for the download process).

## Demo
You can see the demo of the project via the gif below.

![Gif of a demo project could not be loaded](https://github.com/nikola1011/yolov3-car-counter/blob/master/demo-yolov3-dlib-window-rec.gif)

## Literature
The projects is based on [Tensor nets](https://github.com/taehoonlee/tensornets), [keras-yolov3 repository](https://github.com/experiencor/keras-yolo3) - find more detailed read on the [blog](https://towardsdatascience.com/object-detection-using-yolov3-using-keras-80bf35e61ce1).
## Dependencies
Install dependencies via pip specified by requirements.txt file.
The code is tested and run with Python 3.7.4 and Python 3.5.6 on Ubuntu 18.04.3 LTS.
(Windows 10 platforms should also be able to run the project).

## Extensibility
You can easily extend this project by changing the classes you are interested in detecting and tracking (see what classes does YoloV3 support and/or change the neural network used by tensornets for better speed/accuracy.
