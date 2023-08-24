FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Global variables for pretrainer
ENV NCCL_DEBUG="INFO"
ENV OMPI_MCA_opal_cuda_support="true"
ENV CONDA_OVERRIDE_GLIBC="2.56"
ENV RANDOM_SEED=0
ENV TORCH_SEED=42


# Copy necessary files
COPY requirements.txt .
# Clone Geneformer repo
# RUN git clone https://huggingface.co/ctheodoris/Geneformer
COPY Geneformer /Geneformer

# Update package repositories and install GCC
RUN apt-get update && apt-get install -y gcc

# Install pip requirements
RUN pip install --no-cache-dir --upgrade pip
RUN python -m pip install -r requirements.txt
# Install Geneformer
RUN python -m pip install ./Geneformer

WORKDIR /pretraining
COPY . /pretraining

ENV ROOT_DIR="/pretraining"

# Creates a non-root user with an explicit UID and adds permission to access the
# /pretraining folder. For more info, please refer to 
# https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    chown -R appuser /pretraining
USER appuser

# Run as appuser to allow reading/writing permissions
RUN mkdir -p ${ROOT_DIR} && chmod 777 ${ROOT_DIR}

CMD ["deepspeed", "--num_gpus", "1", "pretrain.py"]