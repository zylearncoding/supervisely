# coding: utf-8

import os
from os import path as osp
import json

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

COLORS_FILE_NAME = 'colors.json'
ELEMENT_NAME = 'name'
ELEMENT_ID = 'id'


# returns mapping: x (unit16) color -> some (row, col) for each unique color except black
def get_color_to_coordinates(img):
    h, w = img.shape[:2]
    unq, unq_inv, unq_cnt = np.unique(img, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    color_to_index = {unq[i]: indxs[i][0] for i in range(len(unq))}
    color_to_coordinates = {color: (index // w, index % w) for color, index in color_to_index.items() if color != 0}
    return color_to_coordinates


def images_dir(ds_name):
    return osp.join(sly.TaskPaths.DATA_DIR, ds_name, 'image_2')


def instances_dir(ds_name):
    return osp.join(sly.TaskPaths.DATA_DIR, ds_name, 'instance')


def read_colors():
    colors_filenane = osp.join(sly.TaskPaths.DATA_DIR, COLORS_FILE_NAME)
    if osp.isfile(colors_filenane):
        sly.logger.info('Will try to read segmentation colors from provided file.')
        labels = json.load(open(colors_filenane))
    else:
        sly.logger.info('Will use default Kitti (Cityscapes) color mapping.')
        default_filepath = osp.join(osp.dirname(__file__), COLORS_FILE_NAME)
        labels = json.load(open(default_filepath))

    instance_classes = [el[ELEMENT_NAME] for el in labels if el['hasInstances']]
    class_to_color = {el[ELEMENT_NAME]: list(el['color']) for el in labels}
    id_to_class = {el[ELEMENT_ID]: el[ELEMENT_NAME] for el in labels}
    sly.logger.info('Determined {} class(es).'.format(len(class_to_color)),
                    extra={'classes': list(class_to_color.keys())})
    return instance_classes, id_to_class, class_to_color


def read_datasets():
    src_datasets = {}

    ds_names = [x.name for x in os.scandir(sly.TaskPaths.DATA_DIR) if x.is_dir()]
    for ds_name in ds_names:
        imgdir = images_dir(ds_name)
        sample_names = [x.name for x in os.scandir(imgdir) if x.is_file()]
        src_datasets[ds_name] = sample_names
        sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))
    return src_datasets


def generate_annotation(src_img_path, inst_path, id_to_class, class_to_color, classes_collection):
    ann = sly.Annotation.from_img_path(src_img_path)

    if os.path.isfile(inst_path):
        instance_img = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED) # expect uint16
        col2coord = get_color_to_coordinates(instance_img)

        # Some dirty hack to determine class correctly, low byte is unused. (Low byte describe)
        current_color_to_class = {color: id_to_class[int(color // 256)] for color in col2coord.keys()}

        for color, class_name in current_color_to_class.items():
            mask = instance_img == color  # exact match for 1d uint16
            bitmap = sly.Bitmap(mask)

            if not classes_collection.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap,
                                         color=class_to_color.get(class_name, sly.color.random_rgb()))
                classes_collection = classes_collection.add(obj_class)

            ann = ann.add_label(sly.Label(bitmap, classes_collection.get(class_name)))
            instance_img[mask] = 0  # to check missing colors, see below

        if np.sum(instance_img) > 0:
            sly.logger.warn('Not all objects or classes are captured from source segmentation.', extra={})
    return ann, classes_collection


def convert():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    classes_collection = sly.ObjClassCollection()
    instance_classes, id_to_class, class_to_color = read_colors()
    src_datasets = read_datasets()

    skipped_count = 0
    samples_count = 0

    for ds_name, sample_names in src_datasets.items():
        dataset = out_project.create_dataset(ds_name)
        dataset_progress = sly.Progress('Dataset {!r}'.format(ds_name), len(sample_names))

        for name in sample_names:
            try:
                src_img_path = osp.join(images_dir(ds_name), name)
                inst_path = osp.join(instances_dir(ds_name), name)
                ann, classes_collection = generate_annotation(src_img_path, inst_path, id_to_class, class_to_color,
                                                              classes_collection)
                item_name = osp.splitext(name)[0]

                dataset.add_item_file(item_name, src_img_path, ann)
                samples_count += 1

            except Exception as e:
                exc_str = str(e)
                sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                    'exc_str': exc_str,
                    'dataset_name': ds_name,
                    'image_name': name
                })
                skipped_count += 1
            dataset_progress.iter_done_report()

    sly.logger.info('Processed.', extra={'samples': samples_count, 'skipped': skipped_count})
    out_meta = sly.ProjectMeta(obj_classes=classes_collection)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('KITTI_SEM_SEG_IMPORT', main)
