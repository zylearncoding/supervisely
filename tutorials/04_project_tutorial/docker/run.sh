docker build -t 04_project_tutorial:latest . && \
docker run --rm -it --init \
	--runtime=nvidia \
	-v `pwd`/../../../supervisely_lib:/workdir/src/supervisely_lib \
	-v `pwd`/../src:/workdir/src \
	-v `pwd`/../data:/sly_task_data \
	-p 8888:8888 04_project_tutorial:latest bash
