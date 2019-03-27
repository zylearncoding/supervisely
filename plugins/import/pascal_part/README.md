# Pascal Part 


#### Usage steps:
1) Download `Pascal VOC` dataset from [official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). (Or use [direct Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) 

Download annotations from [official site](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html)

2) Unpack archives

3) Directory structure have to be the following:

```

.
├── Annotations_Part
│   ├── 2008_000002.mat
│   ├── 2008_000003.mat
│   └── ...
└── JPEGImages
    ├── 2007_000027.jpg
    ├── 2007_000032.jpg
    └── ...

```

4) Open [Supervisely import](supervise.ly/import) page. Choose `Pascal Part` import plugin.
5) Select all subdirectories (`ImageSets`, `Annotations_Part`) and drag and drop them to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow dataset: `dataset`.

    ![](https://imgur.com/a/2LF9eSU)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://imgur.com/a/tiRYr8N)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://imgur.com/a/htS7tDK)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).




