docker build -t 12_icnet_inference:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../../../nn/icnet/src:/workdir/src/icnet \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 12_icnet_inference:latest bash
