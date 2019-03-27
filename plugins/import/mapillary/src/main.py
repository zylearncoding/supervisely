# coding: utf-8

import os
import cv2
import numpy as np
import json

import supervisely_lib as sly


# returns mapping: x (unit16) color -> some (row, col) for each unique color except black
def get_col2coord(img):
    h, w = img.shape[:2]
    unq, unq_inv, unq_cnt = np.unique(img, return_inverse=True, return_counts=True)
    indexes = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2index = {unq[i]: indexes[i][0] for i in range(len(unq))}
    col2coord = {col: (index // w, index % w) for col, index in col2index.items() if col != 0}
    return col2coord


class ImporterMapillary:

    def __init__(self):
        self.settings = json.load(open(sly.TaskPaths.SETTINGS_PATH))
        self.colors_file = os.path.join(sly.TaskPaths.DATA_DIR, 'config.json')
        self.obj_classes = sly.ObjClassCollection()
        self._read_colors()
        self._read_datasets()

    def _read_datasets(self):
        self.src_datasets = {}
        ds_names = [x for x in os.listdir(sly.TaskPaths.DATA_DIR)
                    if os.path.isdir(os.path.join(sly.TaskPaths.DATA_DIR, x))]
        for ds_name in ds_names:
            imgdir = self._imgs_dir(ds_name)
            sample_names = [os.path.splitext(x)[0]
                            for x in os.listdir(imgdir) if os.path.isfile(os.path.join(imgdir, x))]
            self.src_datasets[ds_name] = sample_names
            sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))

    def _read_colors(self):
        if os.path.isfile(self.colors_file):
            sly.logger.info('Will try to read segmentation colors from provided file.')
            color_info = json.load(open(self.colors_file))
        else:
            sly.logger.info('Will use default Mapillary color mapping.')
            default_filepath = os.path.join(os.path.dirname(__file__), 'colors.json')
            color_info = json.load(open(default_filepath))
        labels = color_info['labels']
        self.instance_classes = [el['readable'] for el in labels if el['instances']]
        self.cls2col = {el['readable']: tuple(el['color']) for el in labels}
        sly.logger.info('Determined {} class(es).'.format(len(self.cls2col)),
                        extra={'classes': list(self.cls2col.keys())})

        self.cls_names = [el['readable'] for el in labels]  # ! from order of labels

    @classmethod
    def _read_img_unchanged(cls, img_path):
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # expect uint16

    @staticmethod
    def _imgs_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'images')

    @staticmethod
    def _segm_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'labels')

    @staticmethod
    def _inst_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'instances')

    def _generate_annotation(self, src_img_path, inst_path):
        ann = sly.Annotation.from_img_path(src_img_path)

        if os.path.isfile(inst_path):
            instance_img = self._read_img_unchanged(inst_path)
            col2coord = get_col2coord(instance_img)
            curr_col2cls = {col: self.cls_names[int(col // 256)]  # some dirty hack to determine class correctly
                            for col, coord in col2coord.items()}

            for color, class_name in curr_col2cls.items():
                mask = instance_img == color  # exact match for 1d uint16
                bitmap = sly.Bitmap(sly.Point(0, 0), data=mask)
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap,
                                         color=self.cls2col.get(class_name, sly.color.random_rgb()))

                if not self.obj_classes.has_key(class_name):
                    self.obj_classes = self.obj_classes.add(obj_class)

                ann = ann.add_label(sly.Label(bitmap, obj_class))
                instance_img[mask] = 0  # to check missing colors, see below

            if np.sum(instance_img) > 0:
                sly.logger.warn('Not all objects or classes are captured from source segmentation.', extra={})
        return ann

    def convert(self):
        out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, self.settings['res_names']['project']), sly.OpenMode.CREATE)

        for ds_name, sample_names in self.src_datasets.items():
            progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names))
            ds = out_project.create_dataset(ds_name)

            for name in sample_names:
                src_img_path = os.path.join(self._imgs_dir(ds_name), name + '.jpg')
                inst_path = os.path.join(self._inst_dir(ds_name), name + '.png')

                if os.path.isfile(src_img_path):
                    ann = self._generate_annotation(src_img_path, inst_path)
                    ds.add_item_file(name, src_img_path, ann=ann)
                progress.iter_done_report()

        out_meta = sly.ProjectMeta(obj_classes=self.obj_classes)
        out_project.set_meta(out_meta)


def main():
    importer = ImporterMapillary()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('MAPILLARY_IMPORT', main)
