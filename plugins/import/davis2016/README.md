# davis2016 Import

#### Usage steps:
1) Download `davis2016` dataset from [official site](https://davischallenge.org/davis2016/code.html).

   * DAVIS-data.zip	


2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	└── DAVIS	
	    ├── Annotations	
	    │   ├── 1080p	
	    │   │   ├── bear	
	    │   │   ├── blackswan	
	    │   │   └── ...	
	    │   └── 480p	
	    │       ├── bear	
	    │       ├── blackswan	
	    │       └── ...	
	    ├── ImageSets	
	    │   ├── 1080p	
	    │   │   ├── train.txt	
	    │   │   ├── trainval.txt	
	    │   │   └── val.txt	
	    │   └── 480p	
	    │       ├── train.txt	
	    │       ├── trainval.txt	
	    │       └── val.txt	
	    └── JPEGImages	
	        ├── 1080p	
	        │   ├── bear	
	        │   ├── blackswan	
	        │   └── ...	
	        └── 480p	
	            ├── bear	
	            ├── blackswan	
	            └── ...	


        
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `davis2016` import plugin.

5) Select all directory (`DAVIS`) and drag and drop it to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `train480p`, `train1080p`, `trainval480p`, `trainval1080p`, `val480p`, `val1080p`.

    ![](https://i.imgur.com/BZEBmCQ.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/6L0yWsc.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/B3NrYjO.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
