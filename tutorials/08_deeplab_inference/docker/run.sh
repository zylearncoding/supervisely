docker build -t 08_deeplab_inference:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../../../nn/deeplab_v3plus/src:/workdir/src/deeplab_v3plus \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 08_deeplab_inference:latest bash
