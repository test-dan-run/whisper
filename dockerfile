# docker build -f dockerfile -t whisper:1.10.0-cuda11.3-${MODEL} .
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
LABEL maintainer "Daniel Leong Shao Jun <git@test-dan-run> <daniel.leongsj@gmail.com>"

ARG MODEL
ARG PORT

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get -qq update && \
    apt-get -qq -y install git wget build-essential libsndfile1 ffmpeg sox && \
    apt-get -qq autoremove && \
    apt-get -qq clean && \
    rm -rf /var/lib/apt/lists/*

ADD . /whisper
WORKDIR /whisper
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

RUN python download_model.py $MODEL
EXPOSE $PORT
