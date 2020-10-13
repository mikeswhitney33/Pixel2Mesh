FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src
RUN apt update && apt install -y libgl1-mesa-glx
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
