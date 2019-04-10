# coding: utf-8

from collections import namedtuple
import os
import re

from legacy_supervisely_lib import sly_logger
from legacy_supervisely_lib import logger, EventType
from legacy_supervisely_lib.dtl_utils.dtl_helper import DtlHelper, DtlPaths
from legacy_supervisely_lib.dtl_utils.image_descriptor import ImageDescriptor
from legacy_supervisely_lib.tasks import task_helpers
from legacy_supervisely_lib.tasks import progress_counter
from legacy_supervisely_lib.utils import json_utils
from legacy_supervisely_lib.utils import logging_utils

import supervisely_lib as sly

from Net import Net

LegacyProjectItem = namedtuple('LegacyProjectItem', [
    'project_name', 'ds_name', 'image_name', 'ia_data', 'img_path', 'ann_path',
])


def make_legacy_project_item(project: sly.Project, dataset, item_name):
    item_name_base, item_ext = os.path.splitext(item_name)
    return LegacyProjectItem(
        project_name=project.name,
        ds_name=dataset.name,
        image_name=item_name_base,
        ia_data = {'image_ext': item_ext},
        img_path=dataset.get_img_path(item_name),
        ann_path=dataset.get_ann_path(item_name)
    )


def check_in_graph():
    helper = DtlHelper()
    net = Net(helper.graph, helper.in_project_metas, helper.paths.results_dir)

    project_name = net.get_save_layer_dest()
    if not re.match("^(?!.*[\/\\|])", project_name):
        raise RuntimeError('Incorrect output project name "{}".'.format(project_name))

    # to ensure validation
    _ = net.get_result_project_meta()
    _ = net.get_final_project_name()
    is_archive = net.is_archive()

    need_download = net.may_require_images()
    return {'download_images': need_download, 'is_archive': is_archive}


def calculate_datasets_conflict_map(helper):
    tmp_datasets_map = {}

    # Save all [datasets : projects] relations
    for _, pr_dir in helper.in_project_dirs.items():
        project = sly.Project(directory=pr_dir, mode=sly.OpenMode.READ)
        for dataset in project:
            projects_list = tmp_datasets_map.setdefault(dataset.name, [])
            projects_list.append(project.name)

    datasets_conflict_map = {}
    for dataset_name in tmp_datasets_map:
        projects_names_list = tmp_datasets_map[dataset_name]
        for project_name in projects_names_list:
            datasets_conflict_map[project_name] = datasets_conflict_map.get(project_name, {})
            datasets_conflict_map[project_name][dataset_name] = (len(projects_names_list) > 1)

    return datasets_conflict_map


def main():
    task_helpers.task_verification(check_in_graph)

    logger.info('DTL started')
    helper = DtlHelper()

    try:
        net = Net(helper.graph, helper.in_project_metas, helper.paths.results_dir)
        helper.save_res_meta(net.get_result_project_meta())
        datasets_conflict_map = calculate_datasets_conflict_map(helper)
    except Exception as e:
        logger.error("Error occurred on DTL-graph initialization step!")
        raise e

    # is_archive = net.is_archive()
    results_counter = 0
    for pr_name, pr_dir in helper.in_project_dirs.items():
        project = sly.Project(directory=pr_dir, mode=sly.OpenMode.READ)
        progress = progress_counter.progress_counter_dtl(pr_name, project.total_items)
        for dataset in project:
            for item_name in dataset:
                try:
                    img_desc = ImageDescriptor(
                        make_legacy_project_item(project, dataset, item_name),
                        datasets_conflict_map[project.name][dataset.name])
                    ann = json_utils.json_load(dataset.get_ann_path(item_name))
                    data_el = (img_desc, ann)
                    export_output_generator = net.start(data_el)
                    for res_export in export_output_generator:
                        logger.trace("image processed", extra={'img_name': res_export[0][0].get_img_name()})
                        results_counter += 1
                except Exception as e:
                    extra = {
                        'project_name': project.name,
                        'ds_name': dataset.name,
                        'image_name': item_name,
                        'exc_str': str(e),
                    }
                    logger.warn('Image was skipped because some error occurred', exc_info=True, extra=extra)
                progress.iter_done_report()

    logger.info('DTL finished', extra={'event_type': EventType.DTL_APPLIED, 'new_proj_size': results_counter})


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly_logger.add_default_logging_into_file(logger, DtlPaths().debug_dir)
    logging_utils.main_wrapper('DTL', main)

