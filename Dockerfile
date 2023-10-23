FROM ubuntu:focal

RUN apt update
RUN apt install -y git python3 python3-pip
RUN python3 -m pip install --upgrade-strategy eager install optimum[onnxruntime]
WORKDIR /root
ADD whisper.py .

CMD [ "/bin/bash" ]