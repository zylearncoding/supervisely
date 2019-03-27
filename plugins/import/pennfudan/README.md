# PennFudan Import

#### Usage steps:
1) Download `PennFudan_Segmentation` dataset from [official site](http://www.cis.upenn.edu/~jshi/ped_html/).

   * PennFudanPed.zip
		* subdir PNGImages(170 images)
        * subdir PedMasks(170 annotations)	


2) Unpack archive

3) Directory structure have to be the following:

```text
	.	
	└── PennFudanPed	
	    ├── PedMasks	
	    │   ├── FudanPed00001_mask.png	
	    │   ├── FudanPed00002_mask.png	
	    │   └── ...	
	    └── PNGImages	
	        ├── FudanPed00001.png	
	        ├── FudanPed00002.png	
	        └── ...	
	
```

4) Open [Supervisely import](supervise.ly/import) page. Choose `PennFudan_Segmentation` import plugin.
5) Select all directory (`PennFudanPed`) and drag and drop it to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow dataset: `dataset`.

    ![](https://i.imgur.com/uFtSypU.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/A058dj1.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/4uf2Y1H.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
