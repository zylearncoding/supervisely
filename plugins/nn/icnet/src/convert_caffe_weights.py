# coding: utf-8
import argparse

import torch

import icnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-weights', required=True)
    parser.add_argument('--out-weights', required=True)
    parser.add_argument('--bn', default=False, type=bool)
    args = parser.parse_args()

    model = icnet.ICNet(version='cityscapes', is_batchnorm=args.bn)
    model.load_caffe_weights(args.in_weights)
    model.eval()
    torch.save(model.state_dict(), args.out_weights)
