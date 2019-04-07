# Freiburg Import

#### Usage steps:
1) Download `Freiburg` dataset from [official site](https://lmb.informatik.uni-freiburg.de/resources/datasets/PartSeg.html).

   * Sitting.tar.gz	

2) Unpack archive

3) Directory structure have to be the following:

```
	.	
	└── Sitting	
	    ├── img	
	    │   ├── image0.jpg	
	    │   ├── image1.jpg	
	    │   └── ...	
	    └── masks	
	        ├── image0.mat	
	        ├── image1.mat	
	        └── ...	
	
```

4) Open [Supervisely import](supervise.ly/import) page. Choose `Freiburg` import plugin.

5) Select all directory (`dataset`) and drag and drop it to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/A5RMjgY.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/qbARWsA.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/WrVyPqD.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
