FROM tensorflow/tensorflow:1.3.0-gpu-py3

COPY scripts /usr/src/scripts
COPY requirements.txt /usr/src/requirements.txt
WORKDIR /usr/src


RUN pip install --upgrade pip
RUN pip install moderngl moderngl-window tflearn numpy==1.16 pillow==5 pyrr pywavefront
RUN apt update && apt install -y libgl1-mesa-dev libx11-dev xvfb && rm -rf /var/lib/apt/lists/*
ENV DISPLAY=:99.0
RUN Xvfb :99 -screen 0 640x480x24 &
CMD ["./scripts/start-screen.sh"]
