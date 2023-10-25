#FROM ubuntu:focal
FROM dustynv/l4t-pytorch:r35.2.1

RUN apt update
RUN apt install -y git python3 python3-pip git unzip zip
RUN python3 -m pip install --upgrade-strategy eager install optimum[onnxruntime-gpu]
RUN python3 -m pip install joblib soundfile librosa scipy
WORKDIR /root
RUN git clone https://github.com/gdesimone97/WhisperOPT
CMD [ "/bin/bash" ]