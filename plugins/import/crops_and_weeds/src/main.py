# coding: utf-8

import os
from os.path import join

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.dtl_utils import dtl_paths
from supervisely_lib.imaging import color as color_utils
from supervisely_lib.figure.figure_bitmap import FigureBitmap
from supervisely_lib.io.json import dump_json_file, load_json_file
from supervisely_lib.project.annotation import Annotation
from supervisely_lib.project.project_structure import ProjectStructure, ProjectFS
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.tasks import progress_counter


class ImporterCropsWeeds:
    def _define_classes(self):
        self.classes = ('weed', 'crop', 'neutral')

    def __init__(self):
        task_paths = dtl_paths.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = load_json_file(task_paths.settings_path)

        if len(task_paths.project_dirs) > 1:
            raise RuntimeError('The project should consist of only one folder.')

        self.dataset_dir = task_paths.project_dirs[0]
        self._define_classes()

    def _get_ann(self, dataset_dir, image_path, project_meta: ProjectMeta):
        figures = []
        mask_color = cv2.imread(join(dataset_dir, 'annotations', image_path.replace('image', 'annotation')))[: ,: ,::-1]

        mask_gt = cv2.imread(join(dataset_dir, 'masks', image_path.replace('image', 'mask')))[:, :, 0]
        crop_mask = np.all(mask_color == [0, 255, 0], axis=2)
        weed_mask = np.all(mask_color == [255, 0, 0], axis=2)
        neutral_mask = np.logical_and((mask_gt == 0), np.all(mask_color == [0, 0, 0], axis=2))
        figures.extend(FigureBitmap.from_mask('weed', (0, 0), weed_mask))
        figures.extend(FigureBitmap.from_mask('crop', (0, 0), crop_mask))
        figures.extend(FigureBitmap.from_mask('neutral', (0, 0), neutral_mask))
        ann = Annotation.new_with_objects(mask_color.shape[:2], figures, project_meta)
        return ann

    def _convert_sample(self, sample_info, project_meta: ProjectMeta):
        image_name = sample_info.image_name
        ext = sample_info.ia_data['image_ext']
        src_image_path = join(self.dataset_dir, 'images', image_name + ext)

        sly.fs.copy_file(src_image_path, sample_info.img_path)
        ann = self._get_ann(self.dataset_dir, image_name + ext, project_meta)
        packed_ann = ann.pack()
        dump_json_file(packed_ann, sample_info.ann_path)

    def _get_images_pathes(self):
        images_list = os.listdir(join(self.dataset_dir, 'images'))
        return images_list

    def convert(self):
        images_pathes = self._get_images_pathes()
        dataset_name = os.path.basename(os.path.normpath(self.dataset_dir))
        out_pr = ProjectStructure(self.settings['res_names']['project'])

        for image_fp in images_pathes:
            image_ext = os.path.splitext(image_fp)[1]
            image_name = os.path.splitext(image_fp)[0]
            dt = {
                "image_ext": image_ext
            }
            out_pr.add_item(dataset_name, image_name, dt)

        out_pr_fs = ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        res_meta = ProjectMeta()
        for class_name in self.classes:
            # TODO: Fix it line by new meta and object classes
            res_meta.classes.add(
                {'title': class_name, 'shape': 'bitmap', 'color': color_utils.rgb2hex(color_utils.random_rgb())})
        res_meta.to_dir(out_pr_fs.project_path)

        progress = progress_counter.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for sample_info in out_pr_fs:
            self._convert_sample(sample_info, res_meta)
            progress.iter_done_report()


def main():
    importer = ImporterCropsWeeds()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('CROPS_WEEDS_IMPORT', main)
