# DICOM Import

DICOM (Digital Imaging and Communications in Medicine) is a standard for handling, storing, printing, and transmitting information in medical imaging.

#### Usage steps:

1) Given organization structure of DICOM files have to be the following:

```
.
├── folder_1
│   ├── file_1
│   ├── ...
│   ├── file_K
│
└── folder_N
    ├── file_1
    ├── ...
    ├── file_M

```

or

```
.
├── file_1
├── file_2
├── ...
└── file_Z

```

**Actually files and folders names can be arbitrary.**

2) Open [Supervisely Import](supervise.ly/import) page. Choose `DICOM` import plugin.

3) Define import settings or select from exist patterns. 
    
`DICOM` import plugin allow to extract meta-information from `DICOM` files. You can choose information fields (for example `Modality`), which you need to save to supervisely project as `tags` information:

```json
{
  "tags": [
    "Manufacturer",
    "ManufacturerModelName",
    "Modality"
  ]
}
```

5) Select `DICOM` files (or folders which contains `DICOM` files) and drag and drop them to browser.

6) Define new project name and click on `START IMPORT` button.

7) After import task finish, you can view the project:

![](https://i.imgur.com/bJRfm9y.png)
    
8) See few examples below. As you can see `tags` annotations describes meta-information from files according to configuration script:

![](https://i.imgur.com/mmBnpyj.jpg)

![](https://i.imgur.com/iZAu9KB.jpg)

## Notes:
* Supervisely [import documentation](https://docs.supervise.ly/import/).