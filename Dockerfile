#FROM ubuntu:focal
FROM dustynv/l4t-pytorch:r35.2.1

RUN apt update
RUN apt install -y git python3 python3-pip git unzip zip wget curl net-tools
RUN python3 -m pip install --upgrade-strategy eager install optimum[onnxruntime-gpu]
RUN python3 -m pip install joblib soundfile librosa scipy
WORKDIR /root
RUN git clone https://github.com/gdesimone97/WhisperOPT
WORKDIR WhisperOPT
RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-avx512_vnni-no-conv/whisper-large-no_conv.zip
RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-avx512_vnni/whisper-large.zip
RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-ternsorrt/whisper-large-tensorrt.zip
RUN unzip *
RUN rm *.zip

CMD [ "/bin/bash" ]