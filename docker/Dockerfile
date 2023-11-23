FROM docker-registry.qualcomm.com/library/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /partslip

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        cmake \
        curl \
        g++ \
        gcc \
        git \
        libgl1 \
        libglib2.0-0 \
        vim \
        wget \
        zip \
        && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
ARG PYTHON_VERSION=3.9
# Install anaconda (taken from pytorch Dockerfile)
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     conda install -y python=${PYTHON_VERSION} && \
     conda clean -ya

# pytorch
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# for pytorch3d
RUN conda install -c fvcore -c iopath -c conda-forge -c bottler fvcore iopath nvidiacub
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

COPY docker/requirements.yml /tmp/
RUN pip install -r /tmp/requirements.yml

RUN conda install -c conda-forge boost=1.73.0 eigen=3.4.0
# Installing cut-pursuit
RUN TAG="ssp+spg" \
    && git clone --recurse-submodules --shallow-submodules --depth 1 --branch=$TAG https://github.com/loicland/superpoint_graph.git /superpoint_graph \
    && cmake -S /superpoint_graph/partition/cut-pursuit -B /superpoint_graph/partition/cut-pursuit/build -DPYTHON_EXECUTABLE=$(which python)\
    && cmake --build /superpoint_graph/partition/cut-pursuit/build

ENV PYTHONPATH=/superpoint_graph/partition/:/superpoint_graph/partition/cut-pursuit/build/src:$PYTHONPATH

# Installing GLIP (after applying the patch)
COPY docker/ /partslip/docker
RUN git clone https://github.com/microsoft/GLIP.git \
    && cd /partslip/GLIP \
    && git checkout fd52c6361f013e70ae7682d90b3ab3ca2bd5e6bc \
    && git apply /partslip/docker/glip.patch \
    && python setup.py build develop

ADD . /partslip
