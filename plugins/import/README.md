# Import Plugins

This folder contains Import Plugins implemented by developers in Supervisely and researchers in community. The plugins are maintained by their respective authors. To propose a new plugin for inclusion, please submit a pull request.

Import plugin convert custom format to Supervisely format. It allows to works with any dataset in standardized way.

Ready to use plugins:
- `sly_format` - supervisely format
- `links` - plugin allows to add remote images to Supervisely by URL 
- `images`
- `videos`
- `binary_masks`
- `cityscapes`
- `kitti`
- `mapillary`
- `pascal_voc`
- `dicom`

Under active development (they will be finished soon):
- `aberystwyth`
- `Berkeley`
- `cocostuff`
- `cocotext`
- `crops_and_weeds`
- `davis2016`
- `etrims`
- `freiburg`
- `graz50facade`
- `graz50facade`
- `Incidentalscene`
- `Incidentalscene2`
- `MotionSegRecData`
- `parisart`
- `pascal_context`
- `pascal_part`
- `pascal_voc`
- `pennfudan`
- `rangepart`
- `sceneparsing`
- `sceneparsing_2`


# How to create custom import

To create custom Import Plugin we recommend to read: 

1. [How Agent works](../agent/MEADME.md)
2. [How to create plugin](../how_to_create_plugin.md) 

As you can see, your code will be executed within the docker container, directory with all task relevant data will be mounted as volume to `/sly_task_data`. Agent will prepare this task directory with the following structure: 

```
/sly_task_data
├── data
│   └── ...
├── logs
│   └── ...
├── results
└── task_config.json
```

* directory `/sly_task_data/data` will contain all files, that user drag-and-drops to web interface
* directory `/sly_task_data/logs` - technical directory for saving container logs. All container STDOUT will be saved automatically by Agent.
* directory `/sly_task_data/results` - this folder will be empty. Your script have to convert data from folder `/sly_task_data/data` and put results here in Supervisely format
* JSON file `/sly_task_data/task_config.json` has the following structure: 

```json
{
  "res_names": {
    "project": "my_super_project"
  },
  "preset": "images",
  "api_token": "OfaV5z24gEQ7ikv2DiVdYu1CXZhMavU7POtJw2iDtQtvGUux31DUyWTXW6mZ0wd3IRuXTNtMFS9pCggewQWRcqSTUi4EJXzly8kH7MJL1hm3uZeM2MCn5HaoEYwXejKT",
  "server_address": "https://app.supervise.ly/",
  "task_id": 1463,
  "append_to_existing_project": true,
  "options": {"custom fields": "are here"}
}
```

* field `res_names->project` is the name of resulting project, that user types in web interface
* field `preset` the name of plugin (it is a technical information, mostly used for debug purposes)
* field `api_token` api token of user that is running plugin. Most of the plugind do not use Public API directly, but if you would like to implement complex logic (for example, like in import plugin "Links") it may be usefull
* field `server_address` - the same puprose as `api_token`
* field `task_id` - technical field, mostly used for debug purposes
* field `append_to_existing_project` - sometimes it is needed to add some data to already existing project. This flag will be used by agend and your code also may use it if needed
* field `options` - contains any additional data your plugin need. User can put here anything. So most of the existing import plugins do not have any options, but there is an exception. See 'videos' plugin for more details
