# Import Videos 
This plugin allows you to upload videos by cutting them into frames with a given step. It can work with one or few video files. The name of the video file corresponds to the name of the resulting dataset. Supported video formats - **"avi"** and **"mp4"**.

File structure which you can drag and drop should look like this:

```
.
├── video_01.mp4
├── video_02.mp4
└── video_03.avi
```

### Settings config

```json
{
  "skip_frame": 25
}
```

Configuration config of this plugin contains only one field `skip_frame` that is responsible for step between two uploaded frames.

![](https://i.imgur.com/4Ysp5u8.png)

Default `skip_frame` value set to 25, which means that only every 25th frame will be uploaded.


### Example
In this example, we will upload one video with “skip frame” value set to 60 which means that every 2 seconds we will extract a frame (under the conditions that a video recorded with 30 fps).
![](https://i.imgur.com/c4BvQJO.gif)
