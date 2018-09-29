# coding: utf-8

from copy import deepcopy
import supervisely_lib as sly
from Layer import Layer
from random import randint

class CustomLayer(Layer):

    action = 'custom'

    layer_settings = {
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def process(self, data_el):
        img_desc, ann_orig = data_el
        img_wh = ann_orig.image_size_wh
        rect_to_add = sly.Rect(left=img_wh[0] / 2 - 100, top=img_wh[1] / 2 - 100, right=img_wh[0] / 2 + 100,
                               bottom=img_wh[1] / 2 + 100)
        img = img_desc.read_image()
        new_img_desc = img_desc.clone_with_img(img)
        ann = deepcopy(ann_orig)
        new_figures_rect = sly.FigureRectangle.from_rect('rand_rect', ann.image_size_wh, rect_to_add)
        new_figures_poly = sly.FigurePolygon.from_np_points('rand_polygon', img_wh,
                                                            [[1, 10], [200, 150], [938.49, 354.32], [941.38, 361.31],
                                                             [948.61, 364.68], [955.83, 364.2], [749.21, 369.52]], [])
        if len(ann["objects"]) > 0:
            rand_num = randint(0, len(ann["objects"]))
            del ann["objects"][rand_num]
        ann['objects'].extend(new_figures_rect)
        ann['objects'].extend(new_figures_poly)

        yield new_img_desc, ann
