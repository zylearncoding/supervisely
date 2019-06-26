# coding: utf-8

import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

classes_dict = sly.ObjClassCollection()


def read_datasets(lists_dir):
    src_datasets = {}
    if not os.path.isdir(lists_dir):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(lists_dir))
    for dirname in os.listdir(lists_dir):
        now_dir = os.path.join(lists_dir, dirname)
        for filename in os.listdir(now_dir):
            if filename.endswith('.txt'):
                sample_names = []
                ds_name = os.path.splitext(filename)[0]
                file_path = os.path.join(now_dir, filename)
                with open(file_path, "r") as file:
                    all_lines = file.readlines()
                    for line in all_lines:
                        line = line.strip('\n').split(' ')
                        sample_names.append(line[0][12:-4]) # Line example: `/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png`
                src_datasets[ds_name + dirname] = sample_names
        sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    instance_img = sly.image.read(inst_path)
    img_gray = cv2.cvtColor(instance_img, cv2.COLOR_BGR2GRAY)
    _, mask_foreground = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    mask_background = (img_gray == 0)
    class_name = 'background'
    color = [1, 1, 1]
    bitmap = sly.Bitmap(data=mask_background)
    if not classes_dict.has_key(class_name):
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=color)
        classes_dict = classes_dict.add(obj_class)
    ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    mask_foreground = mask_foreground.astype(np.bool)
    if np.any(mask_foreground):
        class_name = 'object'
        color = [255, 255, 255]
        bitmap = sly.Bitmap(data=mask_foreground)
        if not classes_dict.has_key(class_name):
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=color)
            classes_dict = classes_dict.add(obj_class)
        ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
        return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    lists_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'DAVIS/ImageSets')
    imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'DAVIS/JPEGImages')
    inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'DAVIS/Annotations')
    src_datasets = read_datasets(lists_dir)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            src_img_path = os.path.join(imgs_dir, name + '.jpg')
            inst_path = os.path.join(inst_dir, name + '.png')
            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path)
                name = name.replace('/', '_')
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
  sly.main_wrapper('DAVIS_2016', main)

