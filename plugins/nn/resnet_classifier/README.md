
# Residual neural network(ResNet)
Deep residual learning framework for image classification task. Which supports several architectural configurations, allowing to achieve a suitable ratio between the speed of work and quality.

### Description:
- **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Input resolution**: customizable
- **Pretrained**: [ImageNet](http://www.image-net.org/)
- **Work modes**: train, inference, deploy

---

### Architecture
Logical scheme of base building block for ResNet:

<img src="https://i.imgur.com/1BuPsVb.png" width=256>

Architectural configurations for ImageNet. Building blocks are shown in brackets, with the numbers of blocks stacked:
<img src="https://i.imgur.com/MCFUlvY.png" width=600>

### Performance
Results for different ResNet configurations on ImageNet dataset on center crops 224x224.

| Network | Top-1 error | Top-5 error |		
|--|--|--|
| ResNet-18 | 30.24 | 10.92 |
| ResNet-34 | 26.70 | 8.58 |
| ResNet-50 | 23.85 | 7.13 |
| ResNet-101 | 22.63 | 6.44 |
| ResNet-152 | 21.69 | 5.94 |

### Train configuration
_General train configuration available in model presets._ 

Also you can read common training configurations [documentation](https://docs.supervise.ly/neural-networks/configs/train_config/).

- `lr` - initial learning rate.
- `epochs` - the count of training epochs.
- `batch_size` - batch sizes for training (`train`) stage.
- `input_size` - input images dimension `width` and `height` in pixels.
- `gpu_devices` - list of selected GPU devices indexes.
- `data_workers` - how many subprocesses to use for data loading.
- `dataset_tags` - mapping for split data to train (`train`) and validation (`val`) parts by images tags. Images must be tagged by `train` or `val` tags.
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last classification layers. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.
- `val_every` - how often perform validation. Measured in number(float) of epochs. 
- `allow_corrupted_samples` - number of corrupted samples ерфе can be skipped during train(`train`) or validation(`val`)
- `lr_descreasing` - determines the learning rate policy. `patience` - the number of epochs after which learning rate will be decreased, `lr_divisor` - the number learning rate will be divided by.

Full training configuration example:

```json
{
  "dataset_tags": {
    "train": "train",
    "val": "val"
  },
  "batch_size": {
    "train": 64,
    "val": 64
  },
  "data_workers": {
    "train": 8,
    "val": 8
  },
  "input_size": {
    "width": 224,
    "height": 224
  },
  "allow_corrupted_samples": {
    "train": 16,
    "val": 16
  },
  "epochs": 100,
  "val_every": 1,
  "lr": 0.001,
  "lr_decreasing": {
    "patience": 30,
    "lr_divisor": 10
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
      "add_suffix": "_resnet"
    }
  }
}
```
