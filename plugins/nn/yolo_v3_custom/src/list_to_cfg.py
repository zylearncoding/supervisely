# coding: utf-8

import os
import argparse

import supervisely_lib as sly
from supervisely_lib.io.json import dump_json_file
from supervisely_lib.nn.hosted.constants import SETTINGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_file', type=str,
        help='Input class lines.', required=True)
    parser.add_argument(
        '--out_dir', type=str,
        help='Dir to save json.', required=True)
    args = parser.parse_args()
    return args


def construct_detection_classes(names_list):
    name_shape_list = [sly.ObjClass(name=name, geometry_type=sly.Rectangle) for name in names_list]
    return name_shape_list


def main():
    args = parse_args()
    with open(args.in_file) as f:
        lines = f.readlines()
    names_list = [ln for ln in (line.strip() for line in lines) if ln]

    out_classes = sly.ObjClassCollection(items=[sly.ObjClass(name=name, geometry_type=sly.Rectangle)
                                                for name in names_list])

    cls_mapping = {x: idx for idx, x in enumerate(names_list)}
    res_cfg = {
        SETTINGS: {},
        'out_classes': out_classes.to_json(),
        'class_title_to_idx': cls_mapping,
    }

    config_filename = os.path.join(args.out_dir, sly.TaskPaths.MODEL_CONFIG_NAME)
    dump_json_file(res_cfg, config_filename, indent=4)
    print('Done: {} -> {}'.format(args.in_file, config_filename))


if __name__ == '__main__':
    main()
