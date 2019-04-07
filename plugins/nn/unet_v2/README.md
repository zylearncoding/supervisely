# U-Net V2

> The U-Net is a more elegant architecture, the so-called “fully convolutional network”.

The main idea is to supplement a usual contracting network by successive layers, where pooling operations are replaced by upsampling operators. Hence these layers increase the resolution of the output. What’s more, a successive convolutional layer can then learn to assemble a precise output based on this information.

### Description:
- **Paper**: [Fully Convolutional Networks for Semantic Segmentation (2015)](https://arxiv.org/abs/1411.4038)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Input resolution**: customizable
- **Pretrained**: ImageNet
- **Weights size**: ~250 mb
- **Work modes**: train, inference, deploy
- **Usage example**: [Multi-class image segmentation using UNet V2](https://docs.supervise.ly/neural-networks/examples/unet_lemon)

---

### Architecture
<img src="https://i.imgur.com/hNcR2VT.png" width=960/>

The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path.

### Improvements
Our implementation have used `VGG-16` pretrained layers for contractiong path and `Batch Normalization` for improving the performance and stability.

### Train configuration
_Default train configuration available in model presets._ 

Also you can read common training configurations [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config/).

- `lr` - Learning rate.
- `epochs` - the count of training epochs.
- `val_every` - validation peroid by epoch (value `0.5` mean 2 validations per epoch).
- `batch_size` - batch sizes for training (`train`) and validation (`val`) stages.
- `gpu_devices` - list of selected GPU devices indexes.
- `data_workers` - how many subprocesses to use for data loading.
- `dataset_tags` - mapping for split data to train (`train`) and validation (`val`) parts by images tags. Images must be tagged by `train` or `val` tags.
- `special_classes` - objects with specified classes will be interpreted in a specific way. Default class name for `background` is `bg`, default class name for `neutral` is `neutral`. All pixels from `neutral` objects will be ignored in loss function. 
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last layer. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.

Full training configuration example:

```json
{
  "lr": 0.001,
  "epochs": 3,
  "val_every": 0.5,
  "batch_size": {
    "val": 6,
    "train": 12
  },
  "input_size": {
    "width": 256,
    "height": 256
  },
  "gpu_devices": [
    0
  ],
  "data_workers": {
    "val": 0,
    "train": 3
  },
  "dataset_tags": {
    "val": "val",
    "train": "train"
  },
  "special_classes": {
    "neutral": "neutral",
    "background": "bg"
  },
  "weights_init_type": "transfer_learning"
}
```

### Inference configuration

For full explanation see [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config).

**`model`** - group contains unique settings for each model:
 
  * `gpu_device` - device to use for inference. Right now we support only single GPU.

 
**`mode`** - group contains all mode settings:

  *  `name` - mode name defines how to apply NN to image (e.g. `full_image` - apply NN to full image)
   
  *  `model_classes` - which classes will be used, e.g. NN produces 80 classes and you are going to use only few and ignore other. In that case you should set `save_classes` field with the list of interested class names. `add_suffix` string will be added to new class to prevent similar class names with exisiting classes in project. If you are going to use all model classes just set `"save_classes": "__all__"`.


Full image inference configuration example:

```json
{
  "model": {
    "gpu_device": 0
  },
  "mode": {
    "name": "full_image",
    "model_classes": {
      "save_classes": "__all__",
      "add_suffix": "_unet"
    }
  }
}
```