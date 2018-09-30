# Description
This repo contains a Jupyter Notebook that, which demonstrates how to open a supervisely project and draw images with annotations.

# Clone repository
``` 
git clone https://github.com/supervisely/supervisely.git
```

# Preparation with supervisely project
Download project from your account. Then unpack archive to the folder `tutorials/04_project_tutorial/data/project`. Also, you need some empty folders `result`, `tmp` and file `task_settings.json`  For example, `04_project_tutorial` folder will look like this:

```
.
├── data
│   ├── data
│   │   └── project
│   │  		├── Test Project__London
│   │  		└── meta.json
│   ├── result
│   ├── tmp
│   └── task_settings.json
│
├── docker
│   ├── Dockerfile
│   └── run.sh
├── README.md
├── result.png
└── src
    └── 04_project_tutorial.ipynb

```

# How to run
Execute the following commands:

```
cd tutorials/04_project_tutorial/docker
./run.sh
```

to build docker image and run the container. Then, within the container:
``` 
jupyter notebook --allow-root --ip=0.0.0.0
```
Your token will be shown in terminal.
After that, run in browser: 
```
http://localhost:8888/?token=your_token
```

After running `04_project_tutorial`, you get the following results:
![Drivable area segmentation](result.png)

