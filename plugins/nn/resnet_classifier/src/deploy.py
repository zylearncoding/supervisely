# coding: utf-8

import cv2

import supervisely_lib as sly
from supervisely_lib.project.annotation import Annotation
from supervisely_lib.nn.hosted.deploy import ModelDeploy
from inference import ResnetSingleImageApplier


class ResnetAnnotationWrappingApplier(ResnetSingleImageApplier):
    def inference(self, img, ann):
        [cls_title] = super().inference(img, ann)
        res_ann = Annotation.new_with_objects(imsize=img.shape[:2], objects=[])
        res_ann.tags = [cls_title]
        return res_ann


def main():
    model_deploy = ModelDeploy(model_applier_cls=ResnetAnnotationWrappingApplier)
    model_deploy.run()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    sly.main_wrapper('RESNET_SERVICE', main)
