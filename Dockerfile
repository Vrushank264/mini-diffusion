ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update && apt install -y htop python3-dev wget nvidia-container-toolkit

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n diffusion_env python=3.7

COPY . src/

RUN /bin/bash  -c "cd src && source activate diffusion_env && pip install -r requirements.txt"