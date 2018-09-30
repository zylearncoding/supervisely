FROM docker.deepsystems.io/supervisely/nn/faster_rcnn/base:v1.1

##############################################################################
# common
##############################################################################

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
        curl=7.47.0-1ubuntu2.8 \
        ca-certificates=20170717~16.04.1 \
        libjpeg-dev=8c-2ubuntu8 \
        libpng-dev \
        software-properties-common=0.96.20.7 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

##############################################################################
# sly dependencies
##############################################################################

# opencv; other packages are deps & mentioned to fix versions
RUN conda install -y -c menpo \
        opencv=3.4.1 \
        numpy=1.14.3 \
    && conda clean --all --yes


##############################################################################
# Jupyter & matplotlib
##############################################################################
RUN pip install jupyter \
        matplotlib


RUN pip install shapely python-json-logger scikit-image
RUN pip uninstall -y prompt-toolkit
RUN pip install prompt-toolkit
ENV PYTHONPATH /workdir/src/faster_rcnn:$PYTHONPATH
WORKDIR /workdir/src