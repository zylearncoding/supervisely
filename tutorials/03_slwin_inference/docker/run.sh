docker build -t 03_slwin_inference:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../../../nn/unet_v2/src:/workdir/src/unet_src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 03_slwin_inference:latest bash
