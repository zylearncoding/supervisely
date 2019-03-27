# Import Images 
This plugin allows you to upload only images without any annotations. It supports several input file structures: 

### 1.  Flat set of images.
In this case, you can drag and drop one or many images from the same directory. File structure should look like this

```
.
├── img_01.JPG
├── img_02.png
├── img_03.jpeg
└── img_04.jpg
```
Plugin upload all images, create Dataset **ds** and put images there.

### 2. Directories with images.
For this case you have to drag and drop one or few directories with images. Directory name defines Dataset name.
 
```
 .
├── my_folder1
│   ├── img_01.JPG
│   ├── img_02.jpeg
│   └── img_03.png
├── my_folder2
│   ├── img_01.JPG
│   ├── img_02.jpeg
│   └── img_03.png
└── my_folder3
    ├── img_01.JPG
    ├── img_02.jpeg
    └── img_03.png
```

In this example we will drag and drop three directories: `my_folder1`, `my_folder2` and `my_folder3` at a time. As a result we will get project with three datasets with the names of corresponding directories.

### Example 
Example of uploading a flat set of images:
![](https://i.imgur.com/COfEHoM.gif)
