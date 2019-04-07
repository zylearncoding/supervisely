# cocostuff Import

#### Usage steps:
1) Download `cocostuff` dataset from [official site](https://github.com/nightrome/cocostuff).

   * train2017.zip - 118k images	
   * val2017.zip - 5k images	
   * stuffthingmaps_trainval2017.zip:
     * subdir train2017(118k annotations), 
     * subdir val2017(5k annotations)		


2) Unpack archive

3) Directory structure have to be the following:

```	
    .
    ├── labels.txt	
    ├── stuffthingmaps_trainval2017	
    │   ├── train2017	
    │   │   ├── 000000000009.png	
    │   │   ├── 000000000025.png	
    │   │   └── ...	
    │   └── val2017	
    │       ├── 000000000139.png	
    │       ├── 000000000285.png	
    │       └── ...	
    ├── train2017	
    │   ├── 000000000009.jpg	
    │   ├── 000000000025.jpg	
    │   └── ...	
    └── val2017	
        ├── 000000000139.jpg	
        ├── 000000000285.jpg	
        └── ...	
        
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `cocostuff` import plugin.

5) Select all subdirectories (`stuffthingmaps_trainval2017`, `train2017`, `val2017`, `labels.txt`) and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `train2017`, `val2017`.

    ![](https://i.imgur.com/aO5zLLa.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/63cxnYh.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/PmmlFtM.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
