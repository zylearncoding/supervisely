# Mapillary Import

> A diverse street-level imagery dataset with pixel‑accurate and instance‑specific human annotations for understanding street scenes around the world.

#### Usage steps:
1) Download `Mapillary` dataset from [official site](https://www.mapillary.com/)

2) Unpack archive

3) Directory structure have to be the following:

    ```
    .
    ├── config.json
    │ 
    ├── training
    │   ├── images
    │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
    │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
    │   │   ├── ...
    │   ├── instances
    │   │   ├── 0035fkbjWljhaftpVM37-g.png
    │   │   ├── 00qclUcInksIYnm19b1Xfw.png
    │   │   ├── ...
    │   └── labels
    │       ├── 0035fkbjWljhaftpVM37-g.png
    │       ├── 00qclUcInksIYnm19b1Xfw.png
    │       ├── ...
    │ 
    ├── validation
    │   ├── images
    │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
    │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
    │   │   ├── ...
    │   ├── instances
    │   │   ├── 0035fkbjWljhaftpVM37-g.png
    │   │   ├── 00qclUcInksIYnm19b1Xfw.png
    │   │   ├── ...
    │   └── labels
    │       ├── 0035fkbjWljhaftpVM37-g.png
    │       ├── 00qclUcInksIYnm19b1Xfw.png
    │       ├── ...
    │
    └── testing
    ├── images
    │   ├── w6G8WHFnNfiJR-457i0MWQ.jpg
    │   ├── 3q78YjbnUSumHU5n-iYEzA.jpg
    │   ├── ...
    ├── instances
    │   ├── [empty folder]
    └── labels
        ├── [empty folder]
    
    
    ```

4) Open [Supervisely Import](supervise.ly/import) page. Choose `Mapillary` import plugin.
5) Select one or more subdirectories (`training`, `validation`, `testing`) and drag and drop them to browser.    
6) Define new project name and click on `START IMPORT` button.
7) After import task finish, you can view created project:

    ![](https://i.imgur.com/ncvoi9J.jpg)

8)  Also you can find more detailed information about project on `Statistics`, `Classes` and `Tags` tabs:

    ![](https://i.imgur.com/fhJF0j8.png)


## Notes:

* The file `config.json` contains class-color information. If file `config.json` is not exist, will be used internal color-mapping information.
* If you will drag and drop parent directory instead of its content, import will crash.
* Supervisely [import documentation](https://docs.supervise.ly/import/).
