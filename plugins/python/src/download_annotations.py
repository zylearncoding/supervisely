project_name = 'CHANGE_TO_YOUR_INPUT_PROJECT_NAME'

import json
import os

project = api.project.get_info_by_name(WORKSPACE_ID, project_name)
if project is None:
    raise RuntimeError('Project {!r} not found'.format(project_name))

dest_dir = os.path.join(RESULT_ARTIFACTS_DIR, project_name)
sly.fs.mkdir(dest_dir)

meta_json = api.project.get_meta(project.id)
with open(os.path.join(dest_dir, 'meta.json'), 'w') as fout:
    json.dump(meta_json, fout, indent=2)

total_images = 0
for dataset in api.dataset.get_list(project.id):
    ann_dir = os.path.join(dest_dir, dataset.name, 'ann')
    sly.fs.mkdir(ann_dir)

    images = api.image.get_list(dataset.id)
    ds_progress = sly.Progress(
        'Downloading annotations for: {!r}/{!r}'.format(project_name, dataset.name),
        total_cnt=len(images))
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        image_names = [image_info.name for image_info in batch]

        #download annotations in json format
        ann_infos = api.annotation.download_batch(dataset.id, image_ids)
        ann_jsons = [ann_info.annotation for ann_info in ann_infos]

        for image_name, ann_info in zip(image_names, ann_infos):
            with open(os.path.join(ann_dir, image_name + '.json'), 'w') as fout:
                json.dump(ann_info.annotation, fout, indent=2)
        ds_progress.iters_done_report(len(batch))
        total_images += len(batch)

sly.logger.info('Project {!r} has been successfully downloaded'.format(project_name))
sly.logger.info('Total number of images: {!r}'.format(total_images))
