# coding: utf-8

import os
import json
import numpy as np

import supervisely_lib as sly


# returns mapping: (r, g, b) color -> some (row, col) for each unique color except black
def get_col2coord(img):
    img = img.astype(np.int32)
    h, w = img.shape[:2]
    colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
    unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
    col2coord = {(col // (256 ** 2), (col // 256) % 256, col % 256): (indx // w, indx % w)
                 for col, indx in col2indx.items()
                 if col != 0}
    return col2coord


default_classes_colors = {
    'neutral': (224, 224, 192),
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128),
}


class ImporterPascalVOCSegm:

    def __init__(self):
        self.settings = json.load(open(sly.TaskPaths.SETTINGS_PATH, "r"))
        self.lists_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'ImageSets/Segmentation')
        self.imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'JPEGImages')
        self.segm_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'SegmentationClass')
        self.inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'SegmentationObject')
        self.colors_file = os.path.join(sly.TaskPaths.DATA_DIR, 'colors.txt')
        self.with_instances = os.path.isdir(self.inst_dir)
        sly.logger.info('Will import data {} instance info.'.format('with' if self.with_instances else 'without'))

        self._read_datasets()
        self._read_colors()

        self.obj_classes = sly.ObjClassCollection()

    def _read_datasets(self):
        self.src_datasets = {}
        if not os.path.isdir(self.lists_dir):
            raise RuntimeError('There is no directory {}, but it is necessary'.format(self.lists_dir))

        for filename in os.listdir(self.lists_dir):
            if filename.endswith('.txt'):
                ds_name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.lists_dir, filename)
                sample_names = list(filter(None, map(str.strip, open(file_path, 'r').readlines())))
                self.src_datasets[ds_name] = sample_names
                sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))

    def _read_colors(self):
        if os.path.isfile(self.colors_file):
            sly.logger.info('Will try to read segmentation colors from provided file.')
            in_lines = filter(None, map(str.strip, open(self.colors_file, 'r').readlines()))
            in_splitted = (x.split() for x in in_lines)
            # Format: {name: (R, G, B)}, values [0; 255]
            self.cls2col = {x[0]: (int(x[1]), int(x[2]), int(x[3])) for x in in_splitted}
        else:
            sly.logger.info('Will use default PascalVOC color mapping.')
            self.cls2col = default_classes_colors

        sly.logger.info('Determined {} class(es).'.format(len(self.cls2col)),
                        extra={'classes': list(self.cls2col.keys())})
        self.color2class_name = {v: k for k, v in self.cls2col.items()}

    def _get_ann(self, img_path, segm_path, inst_path):

        segmentation_img = sly.image.read(segm_path)

        if inst_path is not None:
            instance_img = sly.image.read(inst_path)
            colored_img = instance_img
            instance_img16 = instance_img.astype(np.uint16)
            col2coord = get_col2coord(instance_img16)
            curr_col2cls = ((col, self.color2class_name.get(tuple(segmentation_img[coord])))
                            for col, coord in col2coord.items())
            curr_col2cls = {k: v for k, v in curr_col2cls if v is not None}  # _instance_ color -> class name
        else:
            colored_img = segmentation_img
            curr_col2cls = self.color2class_name

        ann = sly.Annotation.from_img_path(img_path)

        for color, class_name in curr_col2cls.items():
            mask = np.all(colored_img == color, axis=2)  # exact match (3-channel img & rgb color)

            bitmap = sly.Bitmap(origin=sly.Point(0, 0), data=mask)
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=color)

            if not self.obj_classes.has_key(class_name):
                self.obj_classes = self.obj_classes.add(obj_class)

            ann = ann.add_label(sly.Label(bitmap, obj_class))
            #  clear used pixels in mask to check missing colors, see below
            colored_img[mask] = (0, 0, 0)

        if np.sum(colored_img) > 0:
            sly.logger.warn('Not all objects or classes are captured from source segmentation.')

        return ann

    def convert(self):
        out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, self.settings['res_names']['project']), sly.OpenMode.CREATE)

        for ds_name, sample_names in self.src_datasets.items():
            ds = out_project.create_dataset(ds_name)
            progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names))

            for name in sample_names:
                src_img_path = os.path.join(self.imgs_dir, name + '.jpg')
                segm_path = os.path.join(self.segm_dir, name + '.png')
                inst_path = None

                if self.with_instances:
                    inst_path = os.path.join(self.inst_dir, name + '.png')

                if all((os.path.isfile(x) or (x is None) for x in [src_img_path, segm_path, inst_path])):
                    ann = self._get_ann(src_img_path, segm_path, inst_path)
                    ds.add_item_file(name, src_img_path, ann=ann)
                progress.iter_done_report()

        out_meta = sly.ProjectMeta(obj_classes=self.obj_classes)
        out_project.set_meta(out_meta)


def main():
    importer = ImporterPascalVOCSegm()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('PASCAL_VOC_IMPORT', main)
