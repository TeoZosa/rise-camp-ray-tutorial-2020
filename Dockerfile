#FROM ubuntu:18.04
FROM anyscale/ray:0.8.7

RUN apt-get update && apt-get install -y vim htop curl unzip wget && apt-get clean

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip && ./aws/install && rm awscliv2.zip && rm -rf ./aws

#RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
#RUN yes > bash Anaconda3-2020.07-Linux-x86_64.sh

# We use conda to install these because the image size is smaller
RUN conda install -y faiss-cpu pytorch torchvision -c pytorch && conda clean -y --all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -U ray==1.0
RUN pip install -U pip torch==1.4.0 torchvision==0.5.0 wandb google-api-python-client==1.7.8
# Avoid surprises by pinning since we need to install from source
#RUN pip install git+https://github.com/NVIDIA/apex.git@4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
# Avoid surprises by pinning since we need to install from source
RUN pip install -U git+git://github.com/huggingface/transformers.git@3a7fdd3f5214d1ec494379e7c65b4eb08146ddb0

RUN mkdir -p /root/rise-camp-tutorial

# Copy workspace dir over
COPY . /root/rise-camp-tutorial
