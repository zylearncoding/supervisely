# Description

This repo contains a Jupyter Notebook, which demonstrates how to use dtl transformations for images using supervisely lib.

# Clone repository

`git clone https://github.com/supervisely/supervisely.git`

# Preparation with supervisely project

Download project from your account. Then unpack archive to the folder `tutorials/05_augmentations_tutorial/data/project`. Also, you need some empty folders `result`, `tmp` and file `task_settings.json`. For example, `05_augmentations_tutorial` folder will look like this:

```
.
├── data
│   ├── data
│   │   └── project
│   │  		├── Test Project__London
│   │  		└── meta.json
│   ├── result
│   ├── tmp
│   └── task_settings.json
│
├── docker
│   ├── Dockerfile
│   └── run.sh
├── README.md
├── result.png
└── src
    └── 05_augmentations_tutorial.ipynb

```

# How to run

Execute the following commands:

```
cd tutorials/05_augmentations_tutorial/docker
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

After running `05_augmentations_tutorial`, you get the following results: 
![Result](result.png)