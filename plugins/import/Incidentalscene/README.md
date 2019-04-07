# Incidental Import

#### Usage steps:
1) Download `Incidental` dataset from [official site](http://rrc.cvc.uab.es/?ch=4&com=downloads).

   * ch4_training_images.zip - 1k images	
   * ch4_training_localization_transcription_gt.zip - 1k annotations
   * ch4_test_images.zip - 500 images	
   * Challenge4_Test_Task1_GT.zip - 500 annotations	


2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	├── ch4_test_images	
	│   ├── img_1.jpg	
	│   ├── img_2.jpg	
	│   └── ...	
	├── ch4_training_images	
	│   ├── img_1.jpg	
	│   ├── img_2.jpg	
	│   └── ...	
	├── ch4_training_localization_transcription_gt	
	│   ├── gt_img_1.txt	
	│   ├── gt_img_2.txt	
	│   └── ...	
	└── Challenge4_Test_Task1_GT	
	    ├── gt_img_1.txt	
	    ├── gt_img_2.txt	
	    └── ...
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Incidental` import plugin.

5) Select all subdirectories (`ch4_training_images`, `ch4_training_localization_transcription_gt`, `ch4_test_images`, `Challenge4_Test_Task1_GT`) and drag and drop them to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow dataset: `dataset`.

    ![](https://i.imgur.com/f957k9a.png)

8) Datasets samples contains images and `text segmentation` annotations. See few example:

    ![](https://i.imgur.com/O4j3Est.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/psrqe95.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
