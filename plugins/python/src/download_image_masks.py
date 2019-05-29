project_name = 'CHANGE_TO_YOUR_INPUT_PROJECT_NAME'

import os.path
import numpy as np
from supervisely_lib.io.json import dump_json_file

sly.logger.info('DOWNLOAD_PROJECT', extra={'title': project_name})
project_info = api.project.get_info_by_name(WORKSPACE_ID, project_name)
dest_dir = os.path.join(RESULT_ARTIFACTS_DIR, project_name)
sly.download_project(api, project_info.id, dest_dir, log_progress=True)
sly.logger.info('Project {!r} has been successfully downloaded. Starting to render masks.'.format(project_name))

project = sly.Project(directory=dest_dir, mode=sly.OpenMode.READ)
machine_colors = {obj_class.name: [idx, idx, idx] for idx, obj_class in enumerate(project.meta.obj_classes, start=1)}
dump_json_file(machine_colors, os.path.join(dest_dir, 'obj_class_to_machine_color.json'), indent=2)
for dataset in project:
    human_masks_dir = os.path.join(dataset.directory, 'masks_human')
    machine_masks_dir = os.path.join(dataset.directory, 'masks_machine')
    sly.fs.mkdir(human_masks_dir)
    sly.fs.mkdir(machine_masks_dir)
    for item_name in dataset:
        item_paths = dataset.get_item_paths(item_name)
        ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)
        mask_img_name = os.path.splitext(item_name)[0] + '.png'

        # Render and save human interpretable masks.
        raw_img = sly.image.read(item_paths.img_path)
        raw_img_rendered = raw_img.copy()
        ann.draw(raw_img_rendered)
        raw_img_rendered = ((raw_img_rendered.astype(np.uint16) + raw_img.astype(np.uint16)) / 2).astype(np.uint8)
        sly.image.write(os.path.join(human_masks_dir, mask_img_name),
                        np.concatenate([raw_img, raw_img_rendered], axis=1))

        # Render and save machine readable masks.
        machine_mask = np.zeros(shape=ann.img_size + (3,), dtype=np.uint8)
        for label in ann.labels:
            label.geometry.draw(machine_mask, color=machine_colors[label.obj_class.name])
        sly.image.write(os.path.join(machine_masks_dir, mask_img_name), machine_mask)

sly.logger.info('Finished masks rendering.'.format(project_name))
