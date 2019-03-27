# Incidental part2 Import

#### Usage steps:
1) Download `Incidental part2` dataset from [official site](http://rrc.cvc.uab.es/?ch=4&com=downloads).

   * ch4_training_word_images_gt.zip
   * ch4_test_word_images_gt.zip



2) Unpack archive

3) Directory structure have to be the following:

```	
	.	
	├── ch4_test_word_images_gt	
	│   ├── coords.txt	
	│   ├── gt.txt	
	│   ├── word_1.png	
	│   ├── word_2.png	
	│   └── ...	
	└── ch4_training_word_images_gt	
	    ├── coords.txt	
	    ├── gt.txt	
	    ├── word_1.png	
	    ├── word_2.png	
	    └── ...	

      
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `Incidental2` import plugin.
5) Select directory (`ch4_training_word_images_gt`) and drag and drop them to browser. Wait a little bit.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view project and see follow dataset: `dataset`.

    ![](https://i.imgur.com/yHYXjcC.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/NWvK3xC.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/eOH4ymV.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
