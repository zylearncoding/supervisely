# coding: utf-8

import os
from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly


classes_dict = sly.ObjClassCollection()


def read_datasets(ann_file):
    if not os.path.isfile(ann_file):
        raise RuntimeError('There is no file {}, but it is necessary'.format(ann_file))
    data = load_json_file(ann_file)
    src_datasets = {'train': [], 'val': []}
    set_to_image_name = data['imgs']
    for images_info in set_to_image_name.values():
        if images_info['set'] == 'test':
            continue
        src_datasets[images_info['set']].append(str(images_info['id']))
    return src_datasets


def read_coords_text(ann_file):
    data = load_json_file(ann_file)
    image_name_to_polygon = data['anns']
    photo_names = {}
    for polygon_info in image_name_to_polygon.values():
        photo = polygon_info['image_id']
        text_coords = polygon_info['polygon']
        text_coords = list(map(round, text_coords))
        if 'utf8_string' in polygon_info.keys():
            text_coords.append(polygon_info['utf8_string'])
        else:
            text_coords.append('')
        if not photo in photo_names.keys():
            photo_names[photo] = text_coords
        else:
            photo_names[photo].extend(text_coords)
    return photo_names


def get_ann(img_path, coords_text):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    class_name = 'text'
    color = [255, 0, 255]
    len_polygon_points = 9
    for i in range(0, len(coords_text), len_polygon_points):
        line = coords_text[i : i + len_polygon_points]
        text = line[8]
        points = [sly.PointLocation(line[i + 1], line[i]) for i in range(0, 8, 2)]
        polygon = sly.Polygon(exterior=points, interior=[])
        if not classes_dict.has_key(class_name):
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Polygon, color=color)
            classes_dict = classes_dict.add(obj_class)  # make it for meta.json
        ann = ann.add_label(sly.Label(polygon, classes_dict.get(class_name), None, text))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'train2017')
    ann_file = os.path.join(sly.TaskPaths.DATA_DIR, 'COCO_Text.json')
    src_datasets = read_datasets(ann_file)
    photo_to_coords_text = read_coords_text(ann_file)
    NAME_ZERO_PADDING = 12
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            full_img_name = name.zfill(NAME_ZERO_PADDING) + '.jpg'
            src_img_path = os.path.join(imgs_dir, full_img_name)
            if all((os.path.isfile(x) or (x is None) for x in [src_img_path])):
                try:
                    coords_text = photo_to_coords_text[int(name)]
                except KeyError:
                    continue
                ann = get_ann(src_img_path, coords_text)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('cocotext', main)

