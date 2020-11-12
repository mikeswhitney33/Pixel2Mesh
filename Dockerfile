# FROM tensorflow/tensorflow:1.3.0-gpu-py3
FROM tensorflow/tensorflow:latest-gpu

COPY scripts /usr/src/scripts
COPY requirements.txt /usr/src/requirements.txt
WORKDIR /usr/src

RUN apt update && apt install -y git libgl1-mesa-dev libx11-dev xvfb && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install -r requirements.txt

ENV DISPLAY=:99.0
ENV PYTHONPATH=/usr/src
ENV TF_CPP_MIN_LOG_LEVEL=1
RUN Xvfb :99 -screen 0 640x480x24 &
CMD ["./scripts/start-screen.sh"]
