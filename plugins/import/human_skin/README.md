# Human Skin Import


#### Usage steps:
1) Download `Human Skin` dataset from [official site](http://cs-chan.com/project1.htm). 

2) Unpack archive

3) Directory structure have to be the following:

```
 	.	
 	├── Ground_Truth	
 	│   ├── GroundT_FacePhoto	
 	│   │   ├── 0520962400.png	
 	│   │   ├── 06Apr03Face.png	
 	│   │   └── ...	
 	│   └── GroundT_FamilyPhoto	
 	│       ├── 07-c140-12family.png	
 	│       ├── 2007_family.png	
 	│       └── ...	
 	└── Pratheepan_Dataset	
 	    ├── FacePhoto	
 	    │   ├── 0520962400.jpg	
 	    │   ├── 06Apr03Face.jpg	
 	    │   └── ...	
 	    └── FamilyPhoto	
 	        ├── 07-c140-12family.jpg	
 	        ├── 2007_family.jpg	
 	        └── ...	

```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Pascal VOC` import plugin.

5) Select all subdirectories (`Ground_Truth`, `Pratheepan_Dataset`) and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `FacePhoto`, `FamilyPhoto`.

    ![](https://i.imgur.com/GvCtlII.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few examples:

    ![](https://i.imgur.com/BxvV7oY.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/2mgMEBe.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
