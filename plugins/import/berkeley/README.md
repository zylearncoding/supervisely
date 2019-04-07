# Berkeley Import

#### Usage steps:
1) Download `Berkeley` dataset from [official site](https://bdd-data.berkeley.edu/portal.html).

   * bdd100k.zip

2) Unpack archives

3) Directory structure have to be the following:

```
	.	
	└── bdd100k	
	    └── seg	
	        ├── color_labels	
	        │   ├── train	
	        │   │   ├── 0a0a0b1a-7c39d841_train_color.png	
	        │   │   ├── 0a0ba96d-7859aaa6_train_color.png	
	        │   │   └── ...	
	        │   └── val	
	        │       ├── 7d2f7975-e0c1c5a7_train_color.png	
	        │       ├── 7d4a9094-0455564b_train_color.png	
	        │       └── ...	
	        └── images	
	            ├── train	
	            │   ├── 0a0a0b1a-7c39d841.jpg	
	            │   ├── 0a0ba96d-7859aaa6.jpg	
	            │   └── ...	
	            └── val	
	                ├── 7d2f7975-e0c1c5a7.jpg	
	                ├── 7d4a9094-0455564b.jpg	
	                └── ...	

```

4) Open [Supervisely import](supervise.ly/import) page. Choose `Berkeley` import plugin.

5) Select all directory (`bdd100k`) and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/wyRH38K.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/PfuWdaf.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/Pp1vLpi.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
