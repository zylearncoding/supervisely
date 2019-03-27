# coding: utf-8

import os, cv2
import numpy as np
import scipy.io

import supervisely_lib as sly
from supervisely_lib.imaging.color import generate_rgb
from supervisely_lib.io.json import load_json_file

classes_dict = sly.ObjClassCollection()


def read_datasets(inst_dir_trainval):
    src_datasets, sample_names = {}, []
    for file in os.listdir(inst_dir_trainval):
        if file.endswith('.mat'):
            sample_names.append(os.path.splitext(file)[0])
    src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def read_colors(colors_file):
    sly.logger.info('Generate random color mapping.')
    number_class = {}
    pixel_color = {}
    with open(colors_file, "r") as file:
        all_lines = file.readlines()
        for line in all_lines:
            line = line.split('\n')[0].split(':')
            number_class[line[0]] = (line[1][1:])

    default_classes_colors, colors = {}, [(0, 0, 0)]
    for class_name in number_class.values():
        new_color = generate_rgb(colors)
        colors.append(new_color)
        default_classes_colors[class_name] = new_color

    for i, j in number_class.items():
        pixel_color[i] = default_classes_colors[j]

    class_to_color = default_classes_colors
    sly.logger.info('Determined {} class(es).'.format(len(class_to_color)),
                        extra={'classes': list(class_to_color.keys())})
    return number_class, pixel_color


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)

    if inst_path is not None:
        mat = scipy.io.loadmat(inst_path)
        instance_img = mat['LabelMap']
        colored_img = cv2.merge((instance_img, instance_img, instance_img))
        current_color_to_class = {}
        temp = np.unique(instance_img)
        for pixel in temp:
            current_color_to_class[pixel] = number_class[str(pixel)]

        for pixel, class_name in current_color_to_class.items():
            mask = np.all(colored_img == pixel, axis=2)  # exact match (3-channel img & rgb color)
            new_color = pixel_color[str(pixel)]
            bitmap = sly.Bitmap(origin=sly.Point(0, 0), data=mask)

            if not classes_dict.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
                classes_dict = classes_dict.add(obj_class) # make it for meta.json

            ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
            #  clear used pixels in mask to check missing colors, see below
            colored_img[mask] = (0, 0, 0)

        if np.sum(colored_img) > 0:
            sly.logger.warn('Not all objects or classes are captured from source segmentation.')

    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'JPEGImages')
    inst_dir_trainval = os.path.join(sly.TaskPaths.DATA_DIR, 'trainval')
    labels_file_path = os.path.join(sly.TaskPaths.DATA_DIR, 'labels.txt')
    number_class, pixel_color = read_colors(labels_file_path)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(inst_dir_trainval)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger

        for name in sample_names:
            src_img_path = os.path.join(imgs_dir, name + '.jpg')
            inst_path = os.path.join(inst_dir_trainval, name + '.mat')

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
  sly.main_wrapper('PASCAL_CONTEXT_IMPORT', main)
