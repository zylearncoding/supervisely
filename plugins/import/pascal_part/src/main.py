# coding: utf-8

import os
import numpy as np
import scipy.io

from supervisely_lib.imaging.color import generate_rgb
from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly

classes_dict = sly.ObjClassCollection()


def read_datasets(inst_dir_trainval):
    src_datasets, sample_names = {}, []
    for file in os.listdir(inst_dir_trainval):
        if file.endswith('.mat'):
            sample_names.append(os.path.splitext(file)[0])
    src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path):
    global classes_dict
    default_classes_colors = {}
    colors = [(0, 0, 0)]
    ann = sly.Annotation.from_img_path(img_path)

    if inst_path is not None:
        mat = scipy.io.loadmat(inst_path)
        mask = mat['anno']
        all_objects = mask[0][0][1][0]
        class_mask, unique_class_mask = {}, []

        for obj in all_objects:
            object_name, object_mask = obj[0], obj[2]
            class_mask[object_name[0]] = object_mask
            unique_class_mask.append([object_name[0], object_mask])
            if len(obj[3]) > 0:
                all_parts = obj[3][0]
                for part in all_parts:
                    class_mask[part[0][0]] = part[1]
                    unique_class_mask.append([part[0][0], part[1]])

        for class_name in class_mask.keys():
            if class_name not in default_classes_colors:
                new_color = generate_rgb(colors)
                colors.append(new_color)
                default_classes_colors[class_name] = new_color

        for temp in unique_class_mask:
            class_name, cl_mask = temp
            mask = cl_mask.astype(np.bool)
            new_color = default_classes_colors[class_name]
            bitmap = sly.Bitmap(data=mask)

            if not classes_dict.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
                classes_dict = classes_dict.add(obj_class)  # make it for meta.json
                ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'JPEGImages')
    inst_dir_trainval = os.path.join(sly.TaskPaths.DATA_DIR, 'Annotations_Part')
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(inst_dir_trainval)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger

        for name in sample_names:
            src_img_path = os.path.join(imgs_dir, name + '.jpg')
            inst_path = os.path.join(inst_dir_trainval, name + '.mat')

            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()

    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
  sly.main_wrapper('PASCAL_PART_IMPORT', main)




