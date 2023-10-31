#FROM ubuntu:focal
FROM dustynv/l4t-pytorch:r35.2.1

RUN apt update
RUN apt install -y git python3 python3-pip git unzip zip wget curl net-tools
RUN python3 -m pip install joblib soundfile librosa scipy
WORKDIR /root
RUN git clone https://github.com/gdesimone97/WhisperOPT
WORKDIR WhisperOPT
#RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-avx512_vnni-no-conv/whisper-large-no_conv.zip
#RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-avx512_vnni/whisper-large.zip
#RUN wget https://github.com/gdesimone97/WhisperOPT/releases/download/whisper-largev2-onnx-ternsorrt/whisper-large-tensorrt.zip
COPY whisper-large/ ./
COPY whisper-large-no_conv/ ./
COPY whisper-large-tensorrt/ ./
RUN wget https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
RUN python3 -m pip install *.whl
RUN rm *.whl
RUN python3 -m pip install --upgrade-strategy eager install optimum[onnxruntime-gpu]
RUN python3 -m pip install datasets
ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CMD [ "/bin/bash" ]