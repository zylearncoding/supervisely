# MotionSegRecData Import

#### Usage steps:
1) Download `MotionSegRecData` dataset from [official site](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/).

   * 701_StillsRaw_full.zip
   * LabeledApproved_full.zip


2) Unpack archives

3) Directory structure have to be the following:

```text
	.	
	├── 701_StillsRaw_full	
	│   ├── 0001TP_006690.png	
	│   ├── 0001TP_006720.png	
	│   └── ...	
	└── LabeledApproved_full	
	    ├── 0001TP_006690_L.png	
	    ├── 0001TP_006720_L.png	
	    └── ...	


```

4) Open [Supervisely import](supervise.ly/import) page. Choose `MotionSegRecData` import plugin.
5) Select all directories (`701_StillsRaw_full`, `LabeledApproved_full`) and drag and drop them to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/rQbSWE8.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/vuB3ugM.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/CtGNZj6.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
