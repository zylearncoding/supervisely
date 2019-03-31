# coding: utf-8

import os
from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly


classes_dict = sly.ObjClassCollection()


def read_datasets(inst_dir, directory):
    src_datasets = {}
    if not os.path.isdir(inst_dir):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(inst_dir))
    sample_names = []
    for file in os.listdir(inst_dir):
        if file.endswith('.txt'):
            sample_names.append(os.path.splitext(file[3:])[0])
            src_datasets[directory] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    class_name = 'text'
    color = [255, 0, 255]
    if inst_path is not None:
        with open(inst_path, "r") as file:
            all_lines = file.readlines()
            for line in all_lines:
                line = line.strip('\n').split(',')[:9]
                text = line[8]
                if text == '###':
                    text = ''
                line = line[:8]
                try:
                    line = list(map(lambda i: int(i), line))
                except ValueError:
                    line[0] = line[0][1:]
                    line = list(map(lambda i: int(i), line))
                points = [sly.PointLocation(line[i + 1], line[i]) for i in range(0, 8, 2)]
                polygon = sly.Polygon(exterior=points, interior=[])

                if not classes_dict.has_key(class_name):
                    obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Polygon, color=color)
                    classes_dict = classes_dict.add(obj_class)  # make it for meta.json
                ann = ann.add_label(sly.Label(polygon, classes_dict.get(class_name), None, text))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    for directory in ['train', 'test']:
        if directory == 'train':
            imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_training_images')
            inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_training_localization_transcription_gt')
        else:
            imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_test_images')
            inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'Challenge4_Test_Task1_GT')
        src_datasets = read_datasets(inst_dir, directory)
        for ds_name, sample_names in src_datasets.items():
            ds = out_project.create_dataset(ds_name) #make train -> img, ann
            progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
            for name in sample_names:
                src_img_path = os.path.join(imgs_dir, name + '.jpg')
                inst_path = os.path.join(inst_dir, 'gt_' + name + '.txt')

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
    sly.main_wrapper('Incidentalscene', main)

