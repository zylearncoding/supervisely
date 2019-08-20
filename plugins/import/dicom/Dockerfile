ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

RUN pip install -U git+https://github.com/pydicom/pydicom.git@38d490ff49a1de22878c965fcc6dbdf8b486289e
RUN conda install -c conda-forge gdcm=2.6.6

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src