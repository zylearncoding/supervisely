# coding: utf-8

import math
import numpy as np
from enum import Enum
from copy import deepcopy
from skimage.morphology import skeletonize, medial_axis, thin
from legacy_supervisely_lib.figure.figure_line import FigureLine
from legacy_supervisely_lib.project.annotation import Annotation

from Layer import Layer


class TypeOfConnection(Enum):
    Start1Start2 = 0
    Start1End2 = 1
    Start2End1 = 2
    End1End2 = 3


class Point:
    def __init__(self, x=0, y=0, line_id=0):
        self.x = x
        self.y = y
        self.line_id = line_id


class Vector:
    def __init__(self, pFrom, pTo, normalize=False):
        x = pTo.x - pFrom.x
        y = pTo.y - pFrom.y

        if normalize:
            norm = math.sqrt(x * x + y * y)
            if norm > 0:
                self.x = x / norm
                self.y = y / norm
            else:
                self.x = self.y = 0
        else:
            self.x = x
            self.y = y

    def MulTo(self, scalar_value):
        self.x = self.x * scalar_value
        self.y = self.y * scalar_value


def Distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def MinLineDistance(line1, line2):
    p1 = line1[0]
    p2 = line1[-1]

    q1 = line2[0]
    q2 = line2[-1]

    distArr = [Distance(p1, q1), Distance(p1, q2), Distance(p2, q1), Distance(p2, q2)]

    minDist = distArr[0]
    minInd = 0
    for i in range(1, len(distArr)):
        if distArr[i] < minDist:
            minDist = distArr[i]
            minInd = i

    connType = minInd
    return minDist, connType


def GetCentroid(points_list, start, count):
    xSum = 0
    ySum = 0

    for i in range(int(start), int(start + count)):
        xSum += points_list[i].x
        ySum += points_list[i].y

    p = Point()
    p.x = xSum / count
    p.y = ySum / count
    return p


def LineToVectorCentr(line):
    halfCnt = (len(line) + 1) / 2
    p1 = GetCentroid(line, 0, halfCnt)
    p2 = GetCentroid(line, len(line) - halfCnt, halfCnt)

    center = Point()
    center.x = (p1.x + p2.x) / 2
    center.y = (p1.y + p2.y) / 2

    return Vector(p1, p2, True), center


def line2vec(line):
    return (max(line[0].x, line[1].x) - min(line[0].x, line[1].x),
            max(line[0].y, line[1].y) - min(line[0].y, line[1].y))


def ScalarProduct(a, b):
    a = Vector(Point(0, 0), Point(a.x, a.y), True)
    b = Vector(Point(0, 0), Point(b.x, b.y), True)
    return a.x * b.x + a.y * b.y


def cos_angle_between_lines(a, b):
    a = line2vec(a)
    b = line2vec(b)
    l = math.sqrt(a[0] ** 2 + a[1] ** 2) * math.sqrt(b[0] ** 2 + b[1] ** 2)
    sp = a[0] * b[0] + a[1] * b[1]
    return sp / l


def DistFromPointToLine(p, lineVNormalized, lineP):
    vectLinePToP = Vector(lineP, p)
    proectLen = ScalarProduct(lineVNormalized, vectLinePToP)

    prPoint = Point()
    prPoint.x = lineP.x + lineVNormalized.x * proectLen
    prPoint.y = lineP.y + lineVNormalized.y * proectLen
    return Distance(prPoint, p)


def MergeLinesInOrder(lineA, lineB, vectorA):
    resLine = lineA[:]
    aLen = Distance(lineA[0], lineA[-1])
    aStart = lineA[0]


    if Distance(lineB[0], lineA[0]) < aLen and Distance(lineB[0], lineA[-1]) < aLen:
        if Distance(lineB[1], lineA[0]) < aLen and Distance(lineB[1], lineA[-1]) < aLen:
            return None

    def filter_function(t, aStart, vectorA):
        tmpVect = Vector(aStart, t)
        proectLen = ScalarProduct(vectorA, tmpVect)
        return proectLen < aLen

    resLine.extend(list(filter(lambda t: filter_function(t, aStart, vectorA), lineB)))
    return resLine


def dist2(v, w):
    return (v.x - w.x) ** 2 + (v.y - w.y) ** 2


def distToSegmentSquared(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return dist2(p, v)
    t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2
    t = max(0, min(1, t));
    return dist2(p, Point(x=v.x + t * (w.x - v.x), y=v.y + t * (w.y - v.y)))


def dist_to_segment(p, v, w):
    return math.sqrt(distToSegmentSquared(p, v, w))


def merge_lines_at_one_line_if_possible(
        lineA,
        lineB,
        mergeCollinearEps,
        mergeMaxDistBetweenEnds,
        mergeMaxDistOrtho
):
    resLine = None

    a, aCentr = LineToVectorCentr(lineA)
    b, bCentr = LineToVectorCentr(lineB)

    # Distance condition
    minDist, conn = MinLineDistance(lineA, lineB)
    is_good_distance = minDist < mergeMaxDistBetweenEnds

    conn = TypeOfConnection(conn)
    scMulti = ScalarProduct(a, b)

    # Ortho distance condition
    distances = [
        {'d': 0, "p0": lineA[0], "p1": lineB[0]},
        {'d': 0, "p0": lineA[-1], "p1": lineB[0]},
        {'d': 0, "p0": lineA[0], "p1": lineB[-1]},
        {'d': 0, "p0": lineA[-1], "p1": lineB[-1]}
    ]

    for dist in distances:
        dist['d'] = Distance(dist['p0'], dist['p1'])

    distances = sorted(distances, key=lambda obj: obj['d'])
    distances = distances[0]

    if distances['p0'] == lineA[0]:
        d1 = dist_to_segment(lineA[0], lineA[1], distances['p1'])
    else:
        d1 = dist_to_segment(lineA[-2], lineA[-1], distances['p1'])

    if distances['p1'] == lineB[0]:
        d2 = dist_to_segment(lineB[0], lineB[1], distances['p0'])
    else:
        d2 = dist_to_segment(lineB[-2], lineB[-1], distances['p0'])

    d = np.min([d1, d2])
    is_good_ortho_distance = (d <= mergeMaxDistOrtho)

    # Angle condition
    almostCollinear = mergeCollinearEps <= cos_angle_between_lines(lineA, lineB)

    if almostCollinear and is_good_distance and is_good_ortho_distance:
        if conn == TypeOfConnection.Start1End2:
            lineA.reverse()
            lineB.reverse()
            a.MulTo(-1)
            b.MulTo(-1)

        if conn == TypeOfConnection.End1End2:
            lineB.reverse()
            b.MulTo(-1)

        if conn == TypeOfConnection.Start1Start2:
            lineA.reverse()
            a.MulTo(-1)

        resLine = MergeLinesInOrder(lineA, lineB, a)

    return resLine


def merge_lines_collinear(lines_list, mergeCollinearEps, mergeMaxDistBetweenEnds, mergeMaxDistOrtho):
    mergeCollinearEps = math.cos(math.pi / 180.0 * mergeCollinearEps)
    toMerge = lines_list[:]
    resLines = []
    curInd = 0
    nextInd = 1

    while len(toMerge) > 0:

        curLine = toMerge[curInd]

        if len(toMerge) == 1:
            resLines.append(curLine)
            break

        nextLine = toMerge[nextInd]

        lineToAdd = merge_lines_at_one_line_if_possible(curLine,
                                                  nextLine,
                                                  mergeCollinearEps,
                                                  mergeMaxDistBetweenEnds,
                                                  mergeMaxDistOrtho)

        if lineToAdd is not None:
            del toMerge[curInd]
            del toMerge[nextInd - 1]

            toMerge.insert(0, lineToAdd)
            nextInd = 1
            continue
        else:
            nextInd = nextInd + 1
            if nextInd == len(toMerge):
                del toMerge[curInd]
                resLines.append(curLine)
                nextInd = 1

    return resLines


def simple_lines_to_lines(simple_lines):
    result = []
    for index, line in enumerate(simple_lines):
        nline = [Point(x=p[0], y=p[1], line_id=index) for p in line]
        result.append(nline)
    return result


def lines_to_simple_lines(lines):
    result = []
    for line in lines:
        points = [[p.x, p.y] for p in line]
        result.append(points)
    return result


def merge_lines(class_title, image_size_wh, figures, collinear_eps, max_distance_between_ends, max_distance_ortho):

    simple_lines = []

    for line in figures:
        packed = line.pack()
        points = packed['points']['exterior']
        simple_lines.append(points)

    lines = simple_lines_to_lines(simple_lines)
    result_lines = merge_lines_collinear(lines, collinear_eps, max_distance_between_ends, max_distance_ortho)
    result_lines = lines_to_simple_lines(result_lines)

    results = []
    for line in result_lines:
        figure = FigureLine.from_np_points(class_title=class_title,
                                  image_size_wh=image_size_wh,
                                  exterior=line)
        results.extend(figure)

    return results


class MergeLinesLayer(Layer):

    action = 'merge_lines'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["collinear_eps", "max_distance_between_ends", "max_distance_ortho", "lines_class"],
                "properties": {
                    "collinear_eps": {
                        "description_en": u"Сollinear threshold epsilon (in degree).",
                        "description_ru": u"Граничный коэффициент коллинеарности (в градусах)",
                        "type": "number",
                        "minimum": 0,
                        "maximum": 90
                    },
                    "max_distance_between_ends": {
                        "description_en": u"Maximal distance for merging 2 lines ends",
                        "description_ru": u"Максимальная дистанция между концами линий для сращивания",
                        "type": "number",
                        "minimum": 0
                    },
                    "max_distance_ortho": {
                        "description_en": u"Maximal orthogonal distance for merging 2 lines ends",
                        "description_ru": u"Максимальная ортогональная (перпендикулярная) дистанция между концами линий",
                        "type": "number",
                        "minimum": 0
                    },
                    "lines_class": {
                        "description_en": u"Apply merging only for selected class",
                        "description_ru": u"Применить сращивание только к определенному классу линий",
                        "type": "string"
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        img_wh = ann_orig.image_size_wh

        collinear_eps = self.settings['collinear_eps']
        max_distance_between_ends = self.settings['max_distance_between_ends']
        max_distance_ortho = self.settings['max_distance_ortho']
        lines_class = self.settings.get('lines_class')

        all_figures = ann['objects']

        figures_result = []
        figures_for_processing = []

        for figure in all_figures:
            if not isinstance(figure, FigureLine):
                figures_result.append(figure)
            else:
                class_title = figure.class_title

                if lines_class == class_title:
                    figures_for_processing.append(figure)
                else:
                    figures_result.append(figure)


        figures_result.extend(merge_lines(class_title=lines_class,
                                          image_size_wh=img_wh,
                                          figures=figures_for_processing,
                                          collinear_eps=collinear_eps,
                                          max_distance_between_ends=max_distance_between_ends,
                                          max_distance_ortho=max_distance_ortho))

        ann['objects'] = figures_result
        yield img_desc, ann
