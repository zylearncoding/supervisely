# KITTI Import

> The KITTI semantic segmentation benchmark consists of 200 semantically annotated train as well as 200 test images corresponding to the KITTI Stereo and Flow Benchmark 2015.

#### Usage steps:
1) Download `KITTI` dataset from [official site](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015).

2) Unpack archive

3) Directory structure have to be the following:

    ```
    .
    ├── testing
    │   └── image_2
    │       ├── 000000_10.png
    │       ├── ...
    │       └── 000199_10.png
    └── training
        ├── image_2
        │   ├── 000000_10.png
        │   ├── ...
        │   └── 000199_10.png
        ├── instance
        │   ├── 000000_10.png
        │   ├── ...
        │   └── 000199_10.png
        ├── semantic
        │   ├── 000000_10.png
        │   ├── ...
        │   └── 000199_10.png
        └── semantic_rgb
            ├── 000000_10.png
            ├── ...
            └── 000199_10.png
    ```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `KITTI` import plugin.

5) Select `training` subdirectory and drag and drop them to browser. Wait a little bit. 
   
6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project samples:

    ![](https://i.imgur.com/BsqqYcr.jpg)
    
    ![](https://i.imgur.com/PWXbZxJ.jpg)
    
8) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/eQmXvBn.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).