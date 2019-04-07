# Graz50facade Import

#### Usage steps:
1) Download `Graz50facade` dataset from [official site](http://www.vision.ee.ethz.ch/~rhayko/paper/cvpr2012_riemenschneider_lattice/).

   * graz50_facade_dataset.zip	

2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	└── graz50_facade_dataset	
	    └── graz50_facade_dataset	
	        ├── images	
	        │   ├── facade_0_0053403_0053679.png	
	        │   ├── facade_0_0053679_0053953.png	
	        │   └── ...	
	        └── labels_full	
	            ├── facade_0_0053403_0053679.png	
	            ├── facade_0_0053679_0053953.png	
	            └── ...	
       
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Graz50facade` import plugin.

5) Select directory (`graz50_facade_dataset`) and drag and drop it to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/UHmQctX.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/M579Img.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/F6liszu.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
