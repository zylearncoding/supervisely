# coding: utf-8

import os.path as osp
from enum import Enum

from ..figure.fig_classes import FigClasses
from ..utils.os_utils import ensure_base_path
from ..utils.json_utils import *

from legacy_supervisely_lib.project import tags_lib


class ProjectMetaFmt(Enum):
    V1_CLASSES = 1  # old format, list of classes
    V2_META = 2  # curr format, classes, plain tags (for imgs & objs)


_DEFAULT_OUT_FMT = ProjectMetaFmt.V2_META


class ProjectMeta(object):
    fmt_to_fname = {
        ProjectMetaFmt.V1_CLASSES: 'classes.json',
        ProjectMetaFmt.V2_META: 'meta.json',
    }

    # py_container is native Python container like appropriate dict or list
    def __init__(self, project_meta_json=None):
        self.classes = FigClasses()
        self.img_tags = tags_lib.TagMetaCollection([])
        self.obj_tags = tags_lib.TagMetaCollection([])

        if type(project_meta_json) is list:
            self._in_fmt = ProjectMetaFmt.V1_CLASSES
            self.classes = FigClasses(classes_lst=project_meta_json)

        elif type(project_meta_json) is dict:
            fields = ['classes', 'tags_images', 'tags_objects', ]
            for f in fields:
                if f not in project_meta_json.keys():
                    raise RuntimeError('Missing field: %s' % f)
            self._in_fmt = ProjectMetaFmt.V2_META
            self.classes = FigClasses(classes_lst=project_meta_json['classes'])
            self.img_tags = tags_lib.TagMetaCollection(
                tag_metas=[tags_lib.TagMeta(tag_meta_json) for tag_meta_json in project_meta_json['tags_images']])
            self.obj_tags = tags_lib.TagMetaCollection(
                tag_metas=[tags_lib.TagMeta(tag_meta_json) for tag_meta_json in project_meta_json['tags_objects']])

        elif project_meta_json is None:
            self._in_fmt = None  # empty meta

        else:
            raise RuntimeError('Wrong meta object type.')

    @property
    def input_format(self):
        return self._in_fmt

    def update(self, rhs):
        self.classes.update(rhs.classes)
        self.img_tags.update(rhs.img_tags.to_list())
        self.obj_tags.update(rhs.obj_tags.to_list())

    # TODO: rename to to_json()
    def to_py_container(self, out_fmt=_DEFAULT_OUT_FMT):
        if out_fmt == ProjectMetaFmt.V1_CLASSES:
            res = self.classes.py_container
        elif out_fmt == ProjectMetaFmt.V2_META:
            res = {
                'classes': self.classes.py_container,
                'tags_images': self.img_tags.to_json(),
                'tags_objects': self.obj_tags.to_json(),
            }
        else:
            raise NotImplementedError()
        return res

    def to_json_file(self, fpath, out_fmt=_DEFAULT_OUT_FMT):
        ensure_base_path(fpath)
        json_dump(self.to_py_container(out_fmt), fpath)

    def to_json_str(self, out_fmt=_DEFAULT_OUT_FMT):
        res = json_dumps(self.to_py_container(out_fmt))
        return res

    def to_dir(self, dir_path, out_fmt=_DEFAULT_OUT_FMT):
        fpath = self.dir_path_to_fpath(dir_path, out_fmt)
        self.to_json_file(fpath, out_fmt)

    @classmethod
    def dir_path_to_fpath(cls, dir_path, fmt=_DEFAULT_OUT_FMT):
        return osp.join(dir_path, cls.fmt_to_fname[fmt])

    @classmethod
    def from_json_file(cls, fpath):
        return cls(json_load(fpath))

    @classmethod
    def from_json_str(cls, s):
        return cls(json_loads(s))

    @classmethod
    def find_in_dir(cls, dir_path):
        for fmt in ProjectMetaFmt:
            fpath = cls.dir_path_to_fpath(dir_path, fmt=fmt)
            if osp.isfile(fpath):
                return fpath
        return None

    @classmethod
    def from_dir(cls, dir_path):
        fpath = cls.find_in_dir(dir_path)
        if not fpath:
            raise RuntimeError('File with meta not found in dir: {}'.format(dir_path))
        return cls.from_json_file(fpath)
