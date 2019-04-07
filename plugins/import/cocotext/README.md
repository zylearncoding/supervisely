# Cocotext Import

#### Usage steps:
1) Download `Cocotext` dataset from [official site](https://vision.cornell.edu/se3/coco-text-2/#terms-of-use).
Download `coco` foto dataset from [official site](http://cocodataset.org/#download).

   * train2014.zip - 83k images	
   * COCO_Text.json


2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	├── COCO_Text.json	
	└── train2014	
	    ├── 000000000009.jpg	
	    ├── 000000000025.jpg	
	    └── ...	
       
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `cocotext` import plugin.

5) Select directory (`train2014`), COCO_Text.json file and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `train`, `val`.

    ![](https://i.imgur.com/3eAkfB8.png)

8) Datasets samples contains images and `text segmentation` annotations. See few example:

    ![](https://i.imgur.com/3UUOJpN.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/JEhaZbr.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
