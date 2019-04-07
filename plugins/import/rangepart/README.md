# RangePart Import

#### Usage steps:
1) Download `RangePart` dataset from [official site](https://lmb.informatik.uni-freiburg.de/resources/datasets/PartSeg.html).

   * RANGE.tar.gz	

2) Unpack archive

3) Directory structure have to be the following:

```
	.	
	└── RANGE	
	    ├── 0.8	
	    │   ├── d_images	
	    │   │   ├── 0_8_0.jpg	
	    │   │   ├── 0_8_1.jpg	
	    │   │   └── ...	
	    │   └── d_masks	
	    │       ├── 0_8_0.mat	
	    │       ├── 0_8_1.mat	
	    │       └── ...	
	    ├── ...	
	    └── 6.0	
	        ├── d_images	
	        │   ├── 6_0_0.jpg	
	        │   ├── 6_0_1.jpg	
	        │   └── ...	
	        └── d_masks	
	            ├── 6_0_0.mat	
	            ├── 6_0_1.mat	
	            └── ...	

	
```

4) Open [Supervisely import](supervise.ly/import) page. Choose `RangePart` import plugin.

5) Select all directories (`0.8` ... `6.0`) and drag and drop it to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `0.8` ... `6.0`.

    ![](https://i.imgur.com/9F5PfH2.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/dfaV72u.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/d6KpwP8.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
