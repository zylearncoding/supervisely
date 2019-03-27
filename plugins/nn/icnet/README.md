# ICNet

>Image cascade network (ICNet) incorporates multi-resolution branches under proper label guidance to address the challenge of real-time segmentation task. The model yields realtime inference on a single GPU card with decent quality results evaluated on challenging Cityscapes dataset.

### Description:
- **Paper**: [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Input resolution**: customizable
- **Pretrained**: [Cityscapes](https://www.cityscapes-dataset.com/)
- **Work modes**: train, inference, deploy

---

### Architecture
<img src="https://i.imgur.com/mNGmjaQ.png" width=960/>
‘CFF’ stands for cascade feature fusion. Numbers in parentheses are feature map size ratios to the full-resolution input. Operations are highlighted in brackets. The final ×4 upsampling in the bottom branch is only used during testing.

### Performance
<center><img src="https://i.imgur.com/wYGYbGB.png" width=512/></center>

Results on Cityscapes test set with image resolution 1024×2048.

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
- `special_classes` - objects with specified classes will be interpreted in a specific way.Default class name for `neutral` is `neutral`. All pixels from `neutral` objects will be ignored in loss function.
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last classification layers. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.
- `val_every` - how often perform validation. Measured in number(float) of epochs. 
- `allow_corrupted_samples` - number of corrupted samples ерфе can be skipped during train(`train`) or validation(`val`)
- `lr_descreasing` - determines the learning rate policy. `patience` - the number of epochs after which learning rate will be decreased, `lr_divisor` - the number learning rate will be divided by.
- `use_batchnorm` - there are two ways to train ICNet model: using BatchNormalization layers after convolutions or not. If you want to use them select `true`or `false` otherwise.

Full training configuration example:
```json
{
  "dataset_tags": {
    "train": "train",
    "val": "val"
  },
  "batch_size": {
    "train": 8,
    "val": 4
  },
  "data_workers": {
    "train": 4,
    "val": 2
  },
  "input_size": {
    "width": 2049,
    "height": 1025
  },
  "allow_corrupted_samples": {
    "train": 3,
    "val": 0
  },
  "special_classes": {
    "neutral": "neutral"
  },
  "epochs": 10,
  "val_every": 1,
  "lr": 0.001,
  "lr_decreasing": {
    "patience": 5,
    "lr_divisor": 10
  },
  "weights_init_type": "transfer_learning",
  "use_batchnorm": true
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
      "add_suffix": "_icnet"
    }
  }
}
```
