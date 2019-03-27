# coding: utf-8

import os
import json
import supervisely_lib as sly


def find_input_datasets():
    def flat_images_found(dir_path):
        img_fnames = sly.fs.list_files(dir_path, sly.image.SUPPORTED_IMG_EXTS)
        res = len(img_fnames) > 0
        return res

    def collect_ds_names(pr_path):
        subdirs = sly.fs.get_subdirs(pr_path)
        subd_with_paths = [(x, os.path.join(pr_path, x)) for x in subdirs]
        res = list(filter(lambda x: flat_images_found(x[1]), subd_with_paths))
        return res

    if flat_images_found(sly.TaskPaths.DATA_DIR):
        sly.logger.info('Input structure: flat set of images.')
        return [('ds', sly.TaskPaths.DATA_DIR), ]

    in_datasets = collect_ds_names(sly.TaskPaths.DATA_DIR)
    if len(in_datasets) > 0:
        sly.logger.info('Input structure: set of dirs (datasets) with images.', extra={'ds_cnt': len(in_datasets)})
        return in_datasets
    else:
        top_subdirs = sly.fs.get_subdirs(sly.TaskPaths.DATA_DIR)
        if len(top_subdirs) == 1:
            new_in_dir = os.path.join(sly.TaskPaths.DATA_DIR, top_subdirs[0])
            in_datasets = collect_ds_names(new_in_dir)
            if len(in_datasets) > 0:
                sly.logger.info('Input structure: dir with set of dirs (datasets) with images.',
                            extra={'ds_cnt': len(in_datasets)})
                return in_datasets
            else:
                raise RuntimeError('Input directory is empty')
        elif len(top_subdirs) == 0:
            raise RuntimeError('Input directory is empty')
        else:
            raise RuntimeError('Too many subfolders in the input directory')


def convert():
    task_settings = json.load(open(sly.TaskPaths.SETTINGS_PATH, "r"))
    in_datasets = find_input_datasets()

    pr = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, task_settings['res_names']['project']), sly.OpenMode.CREATE)
    for ds_name, ds_path in in_datasets:
        img_paths = sly.fs.list_files(ds_path, sly.image.SUPPORTED_IMG_EXTS)
        sly.logger.info('Dataset {!r} contains {} image(s).'.format(ds_name, len(img_paths)))
        ds = pr.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(img_paths))
        for img_path in img_paths:
            item_name = sly.fs.get_file_name(img_path)
            if ds.item_exists(item_name):
                item_name = item_name + '_' + sly.rand_str(5)
            ds.add_item_file(item_name, img_path)
            progress.iter_done_report()


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('IMPORT_IMAGES', main)
