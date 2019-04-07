# ParisArt Import

#### Usage steps:
1) Download `parisart` dataset from [official site](https://github.com/raghudeep/ParisArtDecoFacadesDataset).

   * ParisArtDecoFacadesDataset-master.zip	


2) Unpack archive

3) Directory structure have to be the following:

```	
	.
	├── ParisArtDecoFacadesDataset-master
	├── images
	│   	├── facade_1.png
	│   	├── facade_2.png
	│   	└── ...
	|
	└── labels
    		├── facade_1.txt
    		├── facade_2.txt
    		└── ...
       
```
 
4) Open [Supervisely import](supervise.ly/import) page. Choose `ParisArt` import plugin.

5) Select directory (`ParisArtDecoFacadesDataset-master`) and drag and drop it to browser. Wait a little bit.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view project and see follow datasets: `dataset`.

    ![](https://i.imgur.com/VsOkWT8.png)

8) Datasets samples contains images and `instance segmentation` annotations. See few example:

    ![](https://i.imgur.com/muNw9fm.png)
    

9) On `Statistics` project tab you can get more information, for example - class distribution:

    ![](https://i.imgur.com/bGwsLg3.png)
    
## Notes:
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
