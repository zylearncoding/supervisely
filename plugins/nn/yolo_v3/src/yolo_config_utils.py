# coding: utf-8
from collections import namedtuple


MODEL_CFG = 'model.cfg'

CONVOLUTIONAL_SECTION = 'convolutional'
NET_SECTION = 'net'
YOLO_SECTION = 'yolo'


ConfigSection = namedtuple('ConfigSection', ['name', 'data'])


def read_config(file_path):
    sections = []
    current_section = ConfigSection(None, [])
    with open(file_path, 'r') as fin:
        for line in fin:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                sections.append(current_section)
                current_section = ConfigSection(stripped[1:(-1)], [])
            elif stripped.startswith('#') or '=' not in stripped:
                current_section.data.append([stripped])
            else:
                name, value = stripped.split('=', maxsplit=1)
                current_section.data.append([name, value])
    sections.append(current_section)
    return sections


def config_item_to_str(data_item):
    if len(data_item) == 1:
        return data_item[0]
    elif len(data_item) == 2:
        return '{}={}'.format(data_item[0], data_item[1])
    else:
        raise ValueError('Invalid text config data item format: {!r}'.format(data_item))


def write_config(config, file_path):
    with open(file_path, 'w') as fout:
        for section in config:
            if section.name is not None:
                fout.writelines(['[{}]\n'.format(section.name)])
            fout.writelines([config_item_to_str(data_item) + '\n' for data_item in section.data])


def find_data_item(config_section, item_name):
    matching_items = [
        data_item for data_item in config_section.data if len(data_item) == 2 and data_item[0] == item_name]
    if len(matching_items) == 1:
        return matching_items[0]
    elif len(matching_items) == 0:
        return None
    else:
        raise KeyError(
            'Duplicate dayta items found in config section {!r} for name {!r}: {!r}'.format(
                config_section.name, item_name, matching_items))


def replace_config_section_values(config_section: ConfigSection, updates: dict):
    updates_mutable = updates.copy()
    for data_item in config_section.data:
        if len(data_item) == 2:
            item_name = data_item[0]
            if item_name in updates:
                update_value = updates_mutable.pop(item_name, None)
                if update_value is None:
                    raise ValueError('Duplicate entries found in the config section {!r} for option {!r}.'.format(
                        config_section.name, item_name))
                data_item[1] = update_value
    config_section.data.extend([name, value] for name, value in updates_mutable.items())
