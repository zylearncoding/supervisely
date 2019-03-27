# coding: utf-8

from ..sly_logger import logger as default_logger
from ..utils.json_utils import json_load
from legacy_supervisely_lib.project import tags_lib


def validate_tag_dublicates_by_name(tags_list, tag_source=''):
    tags_names = [tags_lib.tag_name_from_json(tag) for tag in tags_list]
    if len(set(tags_names)) < len(tags_names):
        raise ValueError('Dublicates in {0} tags: {1}'.format(tag_source, sorted(tags_names)))


def validate_tag_info(tag, tag_source: str):
    if isinstance(tag, str):
        return True

    if isinstance(tag, dict):
        if ('name' in tag) and ('value' in tag):
            return True

    raise ValueError('Wrong {0} tag "{1}" format!'.format(tag_source, tag))


def validate_and_fix_tag_by_meta(tag, tags_meta, tag_source='', logger=default_logger):
    validate_tag_info(tag, tag_source)

    tag_name = tags_lib.tag_name_from_json(tag)
    if tags_meta.get_tag_meta_by_name(tag_name) is None:
        logger.warn('Unable to find {0} tag "{1}" in project meta. Adding new tag to meta...'.format(tag_source,tag))
        tags_meta.update([tags_lib.TagMeta.from_tag_json(tag)])

    tag_meta = tags_meta.get_tag_meta_by_name(tag_name)
    tag_value = tags_lib.tag_value_from_json(tag)
    if not tag_meta.is_valid_value(tag_value):
        if tag_meta.value_type == tags_lib.TagMeta.VALUE_TYPE_ONEOF_STRING:
            logger.warn('Unable to find {0} tag possible value "{1}" in tag meta. Adding to meta...'.format(
                tag_source, tag))
            tag_meta.add_value(tag_value)
        else:
            raise ValueError(
                'Invalid value {} for tag {}. Raw tag: {}. Raw tag meta: {}'.format(tag_value, tag_name, str(tag),
                                                                                    str(tag_meta.to_json())))


def check_and_fix_project_meta(project_meta, project_fs, logger=default_logger):
    project_items = list(project_fs)
    for it in project_items:
        ann = json_load(it.ann_path)
        validate_tag_dublicates_by_name(ann.tags, 'image')

        for tag in ann.tags:
            validate_and_fix_tag_by_meta(tag, project_meta.img_tags, 'image', logger)

        for obj in ann['objects']:
            class_title = obj['classTitle']
            if class_title not in project_meta.classes.unique_names:
                raise RuntimeError('Unable to find objects class "{}" in project meta.'.format(class_title))

            validate_tag_dublicates_by_name(obj['tags'], 'object')
            for tag in obj['tags']:
                validate_and_fix_tag_by_meta(tag, project_meta.obj_tags, 'object', logger)

    return project_meta
