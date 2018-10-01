# Description
This repo contains a Jupyter Notebook that makes inference of tf-ssd model.

# Clone repository
``` 
git clone https://github.com/supervisely/supervisely.git
```

# Preparation with NN weights
Download NN from your account. Then unpack archive to the folder `tutorials/09_ssd_inference/data/model`. For example, `09_ssd_inference` folder will look like this:

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
    └── 09_ssd_inference.ipynb

```

# How to run
Execute the following commands:

```
cd tutorials/09_ssd_inference/docker
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
If you'l have some errors, when applying string `applier = SSDFastApplier(settings)` you need to remove `override_base_feature_extractor_hyperparams: true`
from the model config file.
After running `09_ssd_inference.ipynb`, you get the following results:
![Segmentation](result.png)

