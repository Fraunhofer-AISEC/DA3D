# General TF2 environment
# Check https://www.tensorflow.org/install/source#gpu
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install tensorflow==2.5.*
RUN pip3 install numpy pandas scikit-learn matplotlib tables

# Install additional packages
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install opencv-python tqdm tensorflow_probability==0.13.* tables

WORKDIR /app
