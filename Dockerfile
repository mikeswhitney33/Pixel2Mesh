FROM tensorflow/tensorflow:1.3.0-gpu-py3


WORKDIR /usr/src
RUN pip install scikit-image tflearn==0.3.2
