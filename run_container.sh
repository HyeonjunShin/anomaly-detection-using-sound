sudo docker run -it --rm --name setting --runtime=nvidia --gpus all ubuntu:22.04 /bin/bash

sudo docker run -it --rm --name anomaly_detection --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/code/anomaly-detection-using-sound/:/code/anomaly-detection-using-sound/ ubuntu:22.04 /bin/bash