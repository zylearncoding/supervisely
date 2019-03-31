# coding: utf-8

import os
import numpy as np

from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly


classes_dict = sly.ObjClassCollection()
count_of_colors = 0 #segmentation colors on all foto are different...

def read_datasets(all_ann):
    src_datasets = {}
    for dirname in os.listdir(all_ann):
        sample_names = []
        for file in os.listdir(os.path.join(all_ann, dirname)):
            if file.endswith('.png'):
                sample_names.append(os.path.splitext(file)[0])
        src_datasets[dirname] = sample_names
        sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(dirname, len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, default_classes_colors, default_colors_classes):
    global classes_dict
    global count_of_colors
    ann = sly.Annotation.from_img_path(img_path)
    curr_color_to_class = {}
    if inst_path is not None:
        instance_img = sly.image.read(inst_path)
        instance_img[np.where((instance_img == [0, 0, 0]).all(axis=2))] = [1, 1, 1]
        colored_img = instance_img * 10
        instance_img = instance_img * 10
        unique_colors = np.unique(instance_img.reshape(-1, instance_img.shape[2]), axis=0)
        ann_colors = np.array(unique_colors).tolist()
        for color in ann_colors:
            if not color in default_classes_colors.values():
                default_classes_colors['object{}'.format(count_of_colors)] = color
                default_colors_classes[tuple(color)] = 'object{}'.format(count_of_colors)
                curr_color_to_class[tuple(color)] = 'object{}'.format(count_of_colors)
                count_of_colors += 1
            else:
                curr_color_to_class[tuple(color)] = default_colors_classes[tuple(color)]

    for color, class_name in curr_color_to_class.items():
        mask = np.all(colored_img == color, axis=2)
        bitmap = sly.Bitmap(data=mask)
        if not classes_dict.has_key(class_name):
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=list(color))
            classes_dict = classes_dict.add(obj_class)  # make it for meta.json

        ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'ADEChallengeData2016/images')
    all_ann = os.path.join(sly.TaskPaths.DATA_DIR, 'annotations_instance')
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(all_ann)
    default_classes_colors = {'background': (10, 10, 10)}
    default_colors_classes = {(10, 10, 10): 'background'}

    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        all_img_temp = os.path.join(all_img, ds_name)
        all_ann_temp = os.path.join(all_ann, ds_name)
        for name in sample_names:
            src_img_path = os.path.join(all_img_temp, name + '.jpg')
            inst_path = os.path.join(all_ann_temp, name + '.png')

            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path, default_classes_colors, default_colors_classes)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()

    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
  sly.main_wrapper('Sceneparsing_Instance_Segmentation', main)

