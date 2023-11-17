FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set CXX env var
ENV CXX g++

# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888

RUN mkdir -p /cache
ENV HF_HOME /cache/huggingface

ENV USER robkirk

# Install python dependencies
COPY requirements.txt /code/rlvsil/requirements.txt
WORKDIR /code/rlvsil

RUN pip install -r requirements.txt
RUN conda install pytorch=2.0.1=py3.10_cuda11.7_cudnn8.5.0_0 -c pytorch

COPY . /code/rlvsil/

# Install system dependencies
# RUN apt update && apt install -y git strace curl vim g++ && rm -rf /var/lib/apt/lists/*
