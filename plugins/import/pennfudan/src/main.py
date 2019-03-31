# coding: utf-8

import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


classes_dict = sly.ObjClassCollection()


def read_datasets(all_ann):
    src_datasets = {}
    sample_names = []
    for file in os.listdir(all_ann):
        if file.endswith('.png'):
            sample_names.append(os.path.splitext(file)[0][:-5])
        src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    if inst_path is not None:
        instance_img = sly.image.read(inst_path)
        class_mask = cv2.split(instance_img)[0]
        class_mask = np.where(class_mask != 0, class_mask, 10)
        current_color2class = {}
        unique_pixels = np.unique(class_mask)
        for pixel in unique_pixels:
            current_color2class[pixel] = number_class[pixel]
    ann = sly.Annotation.from_img_path(img_path)
    for pixel, class_name in current_color2class.items():
        new_color = pixel_color[pixel]
        mask = np.where(class_mask == pixel, class_mask, 0)
        mask = mask.astype(np.bool)
        bitmap = sly.Bitmap(data=mask)
        if not classes_dict.has_key(class_name):
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=list(new_color))
            classes_dict = classes_dict.add(obj_class)
        ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'PennFudanPed/PNGImages')
    all_ann = os.path.join(sly.TaskPaths.DATA_DIR, 'PennFudanPed/PedMasks')
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    number_class = {10: 'background', 1: 'object1', 2: 'object2', 3: 'object3', 4: 'object4', 5: 'object5',
                    6: 'object6', 7: 'object7', 8: 'object8'}
    pixel_color = {10: (0, 0, 0), 1: (255, 255, 0), 2: (255, 0, 255), 3: (0, 255, 255), 4: (0, 255, 0), 5: (255, 0, 0),
                   6: (0, 0, 255), 7: (127, 0, 217), 8: (248, 248, 248)}
    src_datasets = read_datasets(all_ann)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            src_img_path = os.path.join(all_img, name + '.png')
            inst_path = os.path.join(all_ann, name + '_mask' + '.png')
            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path, number_class, pixel_color)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('PennFudan', main)

