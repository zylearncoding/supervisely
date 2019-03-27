FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y \
    libopencv-dev=2.4.9.1+dfsg-1.5ubuntu1.1 && \
    ldconfig

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.27.1-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-4.3.27.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.3.27.1-Linux-x86_64.sh

# Python
ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py

ENV PYTHONPATH='/src/:$PYTHONPATH'

RUN conda install -y -c menpo opencv

RUN pip install --no-cache-dir \
        shapely \
        python-json-logger==0.1.8 \
        pybase64 \
        pyclipper \
        cython \
        networkx \
        scikit-image \
        opencv-python

############### java to run pycharm ###############
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get install -y default-jdk && \
    ln -s /usr/lib/jvm/java-7-openjdk-amd64 /jre
ENV JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64


##############################################################################
# Additional project libraries
##############################################################################
RUN pip install --no-cache-dir \
    pandas==0.22.0 \
    grpcio==1.12.1 \
    grpcio-tools==1.12.1 \
    simplejson==3.16.0

RUN pip install requests-toolbelt
RUN pip install PTable
RUN apt-get install -y fonts-noto=20160116-1


ENV LANG C.UTF-8

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

RUN cd /workdir/src/darknet && make

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src
