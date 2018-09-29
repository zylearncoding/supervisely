# coding: utf-8

from copy import deepcopy
import supervisely_lib as sly
from Layer import Layer


class CustomLayer(Layer):

    action = 'custom'

    layer_settings = {
    }
    test_rect = {
        "description": "",
        "bitmap": 'null',
        "tags": [],
        "classTitle": "Test",
        "points":
            {
                "exterior":
                    [[1618.24, 26.06], [1651.1, 56.57], [1681.14, 21.36], [1654.85, 0.0], [1642.18, 0.0],
                     [1618.24, 26.06]],
                "interior": []
            }
    }
    def __init__(self, config):
        Layer.__init__(self, config)

    def requires_image(self):
        return True

    def process(self, data_el):
        new_class_title = 'new-region'



        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh
        rect_to_add = sly.Rect(left=0, top=0, right=100, bottom=100)
        img = img_desc.read_image()

        new_img_desc = img_desc.clone_with_img(img)
        ann = deepcopy(ann_orig)
        new_figures = sly.FigureRectangle.from_rect(new_class_title, ann.image_size_wh, rect_to_add)
        ann['objects'].extend(new_figures)

        yield new_img_desc, ann
