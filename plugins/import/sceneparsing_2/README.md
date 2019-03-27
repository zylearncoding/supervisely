# Sceneparsing_Segmentation Import

#### Usage steps:
1) Download `Sceneparsing_Segmentation` dataset from [official site](http://sceneparsing.csail.mit.edu/).

   * ADEChallengeData2016.zip
	* subdir images
         * subdir training(20k annotations), 
     	 * subdir validation(2k annotations)		
   	* annotations:
     	 * subdir training(20k annotations), 
     	 * subdir validation(2k annotations)		


2) Unpack archive

3) Directory structure have to be the following:

```	
	.
	└── ADEChallengeData2016	
	    ├── annotations	
	    │   ├── training	
	    │   │   ├── ADE_train_00000001.png	
	    │   │   ├── ADE_train_00000002.png	
	    │   │   └── ADE_train_00000003.png	
	    │   └── validation	
	    │       ├── ADE_val_00000001.png	
	    │       ├── ADE_val_00000002.png	
	    │       └── ADE_val_00000003.png	
	    ├── images	
	    │   ├── training	
	    │   │   ├── ADE_train_00000001.jpg	
	    │   │   ├── ADE_train_00000002.jpg	
	    │   │   └── ADE_train_00000003.jpg	
	    │   └── validation	
	    │       ├── ADE_val_00000001.jpg	
	    │       ├── ADE_val_00000002.jpg	
	    │       └── ADE_val_00000003.jpg	
	    └── objectInfo150.txt	
		
	       
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Sceneparsing_Segmentation` import plugin.
5) Select directory (`ADEChallengeData2016`) and drag and drop it to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow datasets: `training`, `validation`.

    ![](https://i.imgur.com/D6ZSg4c.png)

8) Datasets samples contains images and `segmentation` annotations. See few example:

    ![](https://i.imgur.com/bGt3Jjs.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/Rs3hOjf.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
