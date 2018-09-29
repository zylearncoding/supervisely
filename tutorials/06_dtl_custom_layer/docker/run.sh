docker build -t 06_dtl_custom:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../../../dtl/src:/workdir/src/dtl \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 06_dtl_custom:latest bash
