docker build -t 07_faster-rcnn_inference:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../../../nn/faster_rcnn/src:/workdir/src/faster_rcnn \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 07_faster-rcnn_inference:latest bash
