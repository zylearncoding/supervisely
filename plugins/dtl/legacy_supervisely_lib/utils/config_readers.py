# coding: utf-8

import random
import collections

from legacy_supervisely_lib.figure.rectangle import Rect


# performs updating like lhs.update(rhs), but operates recursively on nested dictionaries
def update_recursively(lhs, rhs):
    for k, v in rhs.items():
        if isinstance(v, collections.Mapping):
            lhs[k] = update_recursively(lhs.get(k, {}), v)
        else:
            lhs[k] = v
    return lhs


# settings fmt from export CropLayer
# TODO remove shift_inside from public api. Instead, have two named versions:
#  padded_rect_from_bounds, cropped_rect_from_bounds
def rect_from_bounds(padding_settings, img_w, img_h, shift_inside=True):
    def get_padding_pixels(raw_side, dim_name):
        side_padding_settings = padding_settings.get(dim_name)
        if side_padding_settings is None:
            padding_pixels = 0
        elif side_padding_settings.endswith('px'):
            padding_pixels = int(side_padding_settings[:-len('px')])
        elif side_padding_settings.endswith('%'):
            padding_fraction = float(side_padding_settings[:-len('%')])
            padding_pixels = int(raw_side * padding_fraction / 100.0)
        else:
            raise ValueError(
                'Unknown padding size format: {}. Expected absolute values as "5px" or relative as "5%"'.format(
                    side_padding_settings))

        if shift_inside:
            padding_pixels *= -1
        return padding_pixels

    def get_padded_side(raw_side, l_name, r_name):
        l_bound = -get_padding_pixels(raw_side, l_name)
        r_bound = raw_side + get_padding_pixels(raw_side, r_name)
        return l_bound, r_bound

    left, right = get_padded_side(img_w, 'left', 'right')
    top, bottom = get_padded_side(img_h, 'top', 'bottom')
    return Rect(left, top, right, bottom)


# @TODO: support float percents?
# returns rect with ints
def random_rect_from_bounds(settings_dct, img_w, img_h):
    def rand_percent(p_name):
        perc_dct = settings_dct[p_name]
        the_percent = random.uniform(perc_dct['min_percent'], perc_dct['max_percent'])
        return the_percent

    def calc_new_side(old_side, perc):
        new_side = min(int(old_side), int(old_side * perc / 100.0))
        l_bound = random.randint(0, old_side - new_side)  # including [a; b]
        r_bound = l_bound + new_side
        return l_bound, r_bound

    rand_percent_w = rand_percent('width')
    if not settings_dct.get('keep_aspect_ratio', False):
        rand_percent_h = rand_percent('height')
    else:
        rand_percent_h = rand_percent_w
    left, right = calc_new_side(img_w, rand_percent_w)
    top, bottom = calc_new_side(img_h, rand_percent_h)
    res = Rect(left, top, right, bottom)
    return res
