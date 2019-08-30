# coding: utf-8

import json
import os
import supervisely_lib as sly


def convert():

    task_settings = json.load(open(sly.TaskPaths.TASK_CONFIG_PATH, 'r'))
    try:
        project = sly.Project(sly.TaskPaths.DATA_DIR, sly.OpenMode.READ)
    except FileNotFoundError:
        possible_projects = sly.fs.get_subdirs(sly.TaskPaths.DATA_DIR)
        if len(possible_projects) != 1:
            raise RuntimeError('Wrong input project structure, or multiple projects are passed.')
        project = sly.Project(os.path.join(sly.TaskPaths.DATA_DIR, possible_projects[0]), sly.OpenMode.READ)
    except Exception as e:
        raise e

    sly.logger.info(
        'Project info: {} dataset(s), {} images(s).'.format(len(project.datasets), project.total_items))
    project.validate()

    project.copy_data(sly.TaskPaths.RESULTS_DIR, dst_name=task_settings['res_names']['project'], _use_hardlink=True)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('SLY_FORMAT_IMPORT', main)
