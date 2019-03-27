# Pascal Context 


#### Usage steps:
1) Download `Pascal` dataset from [official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). (Or use [direct Link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar))

Download annotations and labels.txt from [official site](https://cs.stanford.edu/~roozbeh/pascal-context/)


2) Unpack archives

3) Directory structure have to be the following:

```
.
├── JPEGImages
│   ├── 2007_000027.jpg
│   ├── 2007_000032.jpg
│   └── ...
├── labels.txt
└── trainval
    ├── 2008_000002.mat
    ├── 2008_000003.mat
    └── ...

```

4) Open [Supervisely import](supervise.ly/import) page. Choose `Pascal Context` import plugin.
5) Select all subdirectories (`ImageSets`, `trainval`), file labels.txt and drag and drop them to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow dataset: `dataset`.

    ![](https://i.imgur.com/23zor6P.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/aGHO0He.png)


9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/WVMA7aP.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).



