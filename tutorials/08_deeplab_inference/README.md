# Description
This repo contains a Jupyter Notebook that makes inference of deeplabv3 model.

# Clone repository
``` 
git clone https://github.com/supervisely/supervisely.git
```

# Preparation with NN weights
Download NN from your account. Then unpack archive to the folder `tutorials/08_deeplab_inference/data/model`. In our model zoo there are only weights of the model. Config file is in this repo `config.json`. For example, `08_deeplab_inference` folder will look like this:

```
.
├── data
│   ├── img
│   │   └── 00000220.png
│   └── model
│       ├── config.json
│       └── model.pt
├── docker
│   ├── Dockerfile
│   └── run.sh
├── README.md
├── result.png
└── src
    └── 08_deeplab_inference.ipynb

```

# How to run
Execute the following commands:

```
cd tutorials/08_deeplab_inference/docker
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

After running `08_deeplab_inference.ipynb`, you get the following results:
![Segmentation](result.png)

