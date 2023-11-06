FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

RUN apt update
RUN python3 -m pip -U pip
RUN wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install -U --ingnore-installed pyyaml
RUN python3 -m pip install ./*.whl
RUN python3 -m pip install optimum[onnxruntime] accelerate

CMD [ "/bin/bash" ]