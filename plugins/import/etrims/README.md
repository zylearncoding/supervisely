# Etrims Import

#### Usage steps:
1) Download `Etrims` dataset from [official site](http://www.ipb.uni-bonn.de/projects/etrims_db/).

   * etrims-db_v1.zip	

2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	└── etrims-db_v1	
	    ├── annotations	
	    │   ├── 04_etrims-ds	
	    │   │   ├── basel_000003_mv0.png	
	    │   │   ├── basel_000004_mv0.png	
	    │   │   └── ...	
	    │   └── 08_etrims-ds	
	    │       ├── basel_000003_mv0.png	
	    │       ├── basel_000004_mv0.png	
	    │       └── ...	
	    └── images	
	        ├── 04_etrims-ds	
	        │   ├── basel_000003_mv0.jpg	
	        │   ├── basel_000004_mv0.jpg	
	        │   └── ...	
	        └── 08_etrims-ds	
	            ├── basel_000003_mv0.jpg	
	            ├── basel_000004_mv0.jpg	
	            └── ...	
     
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Etrims` import plugin.
5) Select directory (`etrims-db_v1`) and drag and drop it to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow datasets: `04_etrims-ds`, `08_etrims-ds`.

    ![](https://i.imgur.com/xy7imgf.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/LUvsVsJ.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/HS1Q152.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
