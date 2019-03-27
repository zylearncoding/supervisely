# Sceneparsing_Instance_Segmentation Import

#### Usage steps:
1) Download `Sceneparsing_Instance_Segmentation` dataset from [official site](http://sceneparsing.csail.mit.edu/).

   * ADEChallengeData2016.zip
	* subdir images
         * subdir training(20k annotations), 
     	 * subdir validation(2k annotations)		
   * annotations_instance.zip:
     * subdir training(20k annotations), 
     * subdir validation(2k annotations)		


2) Unpack archive

3) Directory structure have to be the following:

```text
.	
├── ADEChallengeData2016	
│   └── images	
│       ├── training	
│       │   ├── ADE_train_00000001.jpg	
│       │   ├── ADE_train_00000002.jpg	
│       │   └── ...		
│       └── validation	
│           ├── ADE_val_00000001.jpg	
│           ├── ADE_val_00000002.jpg	
│           └── ...	
└── annotations_instance		
├── training	
│   ├── ADE_train_00000001.png	
│   ├── ADE_train_00000002.png	
│   └── ...	
└── validation	
├── ADE_val_00000001.png	
├── ADE_val_00000002.png	
└── ...
```

4) Open [Supervisely import](supervise.ly/import) page. Choose `Sceneparsing_Instance_Segmentation` import plugin.
5) Select all directories (`ADEChallengeData2016`, `annotations_instance`) and drag and drop them to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow datasets: `training`, `validation`.

    ![](https://i.imgur.com/kHozJlX.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/Pn373X0.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/wzbniiA.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
