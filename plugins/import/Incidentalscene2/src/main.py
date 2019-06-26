# coding: utf-8

import os
from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly


classes_dict = sly.ObjClassCollection()


def read_datasets(coord_file, dataset):
    src_datasets = {}
    if not os.path.isfile(coord_file):
        raise RuntimeError('There is no file {}, but it is necessary'.format(coord_file))
    sample_names = []
    with open(coord_file, "r") as file:
        all_lines = file.readlines()
        for line in all_lines:
            line = line.strip('\n').split(',')[0]
            if line[0] != 'w':
                line = line[1:]
            sample_names.append(line[:-4])
        src_datasets[dataset] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def read_words(word_file):
    words = {}
    with open(word_file, "r") as file:
        all_lines = file.readlines()
        for line in all_lines:
            word = line.strip('\n').split(',')[1][2:-1]
            name = line.strip('\n').split(',')[0]
            if name[0] != 'w':
                name = name[1:]
            words[name] = word
    return words


def read_coords(coord_file):
    coords = {}
    with open(coord_file, "r") as file:
        all_lines = file.readlines()
        for line in all_lines:
            name = line.strip('\n').split(',')[0]
            line = line.strip('\n').split(',')[1:]
            if name[0] != 'w':
                name = name[1:]
            line = list(map(lambda i: int(i), line))
            coords[name] = line
    return coords


def get_ann(img_path, coords, words):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    class_name = 'text'
    color = [255, 0, 0]
    name = img_path.split('/')[-1]
    line = coords[name]
    points = [sly.PointLocation(line[i + 1], line[i]) for i in range(0, 8, 2)]
    polygon = sly.Polygon(exterior=points, interior=[])
    if not classes_dict.has_key(class_name):
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Polygon, color=color)
        classes_dict = classes_dict.add(obj_class)  # make it for meta.json
    ann = ann.add_label(sly.Label(polygon, classes_dict.get(class_name), None, words[name]))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    for dataset in ['train', 'test']:
        if dataset == 'train':
            imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_training_word_images_gt')
            coord_file = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_training_word_images_gt/coords.txt')
            word_file = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_training_word_images_gt/gt.txt')
        else:
            imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_test_word_images_gt')
            coord_file = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_test_word_images_gt/coords.txt')
            word_file = os.path.join(sly.TaskPaths.DATA_DIR, 'ch4_test_word_images_gt/gt.txt')

        src_datasets = read_datasets(coord_file, dataset)
        words = read_words(word_file)
        coords = read_coords(coord_file)
        for ds_name, sample_names in src_datasets.items():
            ds = out_project.create_dataset(ds_name) #make train -> img, ann
            progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
            for name in sample_names:
                src_img_path = os.path.join(imgs_dir, name + '.png')

                if all((os.path.isfile(x) or (x is None) for x in [src_img_path])):
                    ann = get_ann(src_img_path, coords, words)
                    ds.add_item_file(name, src_img_path, ann=ann)
                progress.iter_done_report()

    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('Incidentalscene2', main)
