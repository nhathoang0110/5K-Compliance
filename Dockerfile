# FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
WORKDIR /model

RUN apt-get update && \
    apt-get install -y git && \
    apt-get -y install cmake && \
    apt-get install -y wget unzip && \
    apt-get install -y --no-install-recommends software-properties-common libboost-all-dev libc6-dbg libgeos-dev python3-dev python3-pip python3-setuptools && \
    apt-get install -y libjpeg-dev zlib1g-dev && \
    apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    rm -rf /var/lib/apt/lists/*

COPY . /model/
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && pip3 install -v --no-cache-dir ./
# RUN cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./
RUN cd ..
# CMD ["bash", "predict.sh"]

