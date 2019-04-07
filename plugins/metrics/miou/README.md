# Intersection over Union (IoU)
### Some theory
Intersection over Union also known as Jaccard index is a statistic used for comparing the similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets:
<center><img src="https://i.imgur.com/8ODEYrM.png"/></center>

Intersection of two sets A and B:
<center><img src="https://i.imgur.com/Bo5aZDS.png" width=256/></center>

Union of two sets A and B:
<center><img src="https://i.imgur.com/9X0BuBw.png" width=256/></center>

IoU metric is often used in computer vision in such tasks as object detection and image segmentation. In these tasks to evaluate IoU, you should calculate area of overlap and area of union between “predicted” and “ground-truth” objects.

<center><img src="https://i.imgur.com/yWlXYXM.png" width=512/></center>

Also it is possible to rewrite IoU formula in terms of true positives(TP), false positives(FP) and false negatives(FN) regions:

<center><img src="https://i.imgur.com/wqWpcLU.png" width=300/></center>

Comparison of IoU values and detection quality:

<center><img src="https://i.imgur.com/CgybuoT.png" width=700/></center>

Here is an example of object detection task, here green rectangle is “ground-truth” object and red is “predicted” object. If we calculate area (in pixels) of their intersection and union, we get IoU value of **0.89**:

<center><img src="https://i.imgur.com/02pyrf1.png" width=700 /></center>


### <center> Supervisely config example</center>

```json
{
  "project_1": "project_1_name",
  "project_2": "project_2_name",
  "classes_mapping": {
    "pr1_cls_name_1": "pr2_cls_name_1",
    "pr1_cls_name_2": "pr2_cls_name_2"
  }
}
```

Here:

1) **“project_1”** - name of the first project involved in IoU evalutaion.
2) **“project_2”** - name of the second project involved in IoU evaluation. In case you want to evaluate metric between objects of the same project, you can simply delete this field from config.
3) **“classes_mapping”** - field matches pairs of objects classes between which IoU will be evaluated.


### <center>Usage example</center>
In this example we will use IoU metric in image segmentation task using Mask-RCNN network. To do this, we take several images with two annotated classes “person” (red) and “car” (green):
<center><img src="https://i.imgur.com/w3atESQ.png" width=700 /></center>
<center><img src="https://i.imgur.com/IJVPdlk.png" width=700 /></center>
<center><img src="https://i.imgur.com/xLTbCmc.png" width=700 /></center>

Next we apply our pretrained on COCO Mask-RCNN model to this pictures. Now we have objects of two predicted classes “person_mask” (brown) and “car_mask” (purple):
<center><img src="https://i.imgur.com/cLCUS3E.png" width=700 /></center>
<center><img src="https://i.imgur.com/HMjYZyx.png" width=700 /></center>
<center><img src="https://imgur.com/Cx5Awvw.png" width=700 /></center>

To run IoU plugin we will use following config:

```json
{
  "project_1": "original_data",
  "project_2": "maskrcnn_inference",
  "classes_mapping": {
    "car": "car_mask",
    "person": "person_mask"
  }
}
```

This plugin calculates IoU values for each pair of classes on each image and then show mean value for all images. Results can be seen in log panel:
<center><img src="https://i.imgur.com/I4fSkAv.png" width=700/></center>


He we can see that model predictions for class “person” have mean IoU for all images ~0.963, for class “car” ~0.977, and average value for two classes ~0.966