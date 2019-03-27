
# Import Binary Masks
This plugin allows you to upload images with annotations in the format of PNG masks. Masks are 3-(1-)channel images containingonly pixels that have the same values in all channels. To map pixels masks with appropriate class you can use specific config:

#### Example 1.
```json
{
	"Cat": 1,
	"Dog": 2,
	"Car": 3
}
```
In this example all pixels in the mask which value **equal to 1** will be combined in one Bitmap figure and will be assigned to the class "Cat", equal to 2 will be assigned to the class "Dog" and equal to 3 will be assigned to the class 3 respectively.

#### Example 2.
```json
{
	"Pets": [1, 2],
	"Car": 3
}
```
In this case all pixels in the mask which value **equal to 1 or 2** will be combined in one Bitmap figure and will be assigned to the class "Pets" and all pixels equal to 3 will be assinged to the class 3 respectively.

#### Example 3.
```json
{
	"objects": "__all__",
}
```
In this case all pixels in the mask which value **greater than 0** will be combined in one Bitmap figure and will be assigned to the class "objects".


Also you don't have to specify any configs, in this case default config will be used:
```json
{
	"untitled": "__all__",
}
```

Images should be in the folder "img" and mask should be in the folder "ann" and should have the same name as the images(but may have a different extension). All images will be  placed in dataset **ds**.

File structure that can be uploaded by this plugin should look like this:
```
.
├── ann
│   ├── image_1.png
│   ├── image_2.png
│   └── image_3.png
└── img
    ├── image_1.png
    ├── image_2.png
    └── image_3.png

```
### Example 
In this example we will upload images with annotated masks of persons. 
![](https://i.imgur.com/BlSp1Pj.gif)
