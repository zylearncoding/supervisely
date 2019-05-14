# coding: utf-8

import json
import os.path
import skvideo.io
import supervisely_lib as sly

DEFAULT_STEP = 25


def convert_video():
    task_settings = json.load(open(sly.TaskPaths.TASK_CONFIG_PATH, 'r'))

    step = DEFAULT_STEP
    if 'step' in task_settings['options']:
        step = int(task_settings['options']['step'])
    else:
        sly.logger.warning('step parameter not found. set to default: {}'.format(DEFAULT_STEP))

    paths = sly.fs.list_files(sly.TaskPaths.DATA_DIR)
    video_paths = []
    for path in paths:
        if sly.video.has_valid_ext(path):
            video_paths.append(path)
        else:
            sly.logger.warning("Video file '{}' has unsupported extension. Skipped. Supported extensions: {}"
                               .format(path, sly.video.ALLOWED_VIDEO_EXTENSIONS))

    if len(video_paths) == 0:
        raise RuntimeError("Videos not found!")

    project_dir = os.path.join(sly.TaskPaths.RESULTS_DIR, task_settings['res_names']['project'])
    project = sly.Project(directory=project_dir, mode=sly.OpenMode.CREATE)
    for video_path in video_paths:
        try:
            video_relpath = os.path.relpath(video_path, sly.TaskPaths.DATA_DIR)
            ds_name = video_relpath.replace('/', '__')
            ds = project.create_dataset(ds_name=ds_name)

            vreader = skvideo.io.FFmpegReader(video_path)

            vlength = vreader.getShape()[0]
            progress = sly.Progress('Import video: {}'.format(ds_name), vlength)

            for frame_id, image in enumerate(vreader.nextFrame()):
                if frame_id % step == 0:
                    img_name = "frame_{:05d}".format(frame_id)
                    ds.add_item_np(img_name + '.png', image)
                progress.iter_done_report()

        except Exception as e:
            exc_str = str(e)
            sly.logger.warn('Input video skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                'exc_str': exc_str,
                'video_file': video_path,
            })


def main():
    convert_video()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('VIDEO_ONLY_IMPORT', main)