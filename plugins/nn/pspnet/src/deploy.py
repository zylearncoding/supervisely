# coding: utf-8

import cv2

import supervisely_lib as sly
from supervisely_lib.nn.hosted.deploy import ModelDeploy
from inference import PSPNetSingleImageApplier


def main():
    model_deploy = ModelDeploy(model_applier_cls=PSPNetSingleImageApplier)
    model_deploy.run()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    sly.main_wrapper('PSPNET_SERVICE', main)
