FROM tensorflow/tensorflow:1.3.0-gpu-py3


WORKDIR /usr/src

RUN pip install --upgrade pip
RUN pip install moderngl moderngl-window tflearn numpy==1.16 pillow==5 pyrr
RUN apt update
RUN apt install -y libgl1-mesa-dev libx11-dev
