from model import sgan
from labels import labels

import cv2 as cv
import numpy as np

import argparse
import glob
import os
import config
import util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--label', type=str, default='airplane')
    parser.add_argument('--edge_path', type=str, default='out/edge.png')
    parser.add_argument('--params_path', type=str, default='params/sgan_model.ckpt')
    return parser.parse_args()

args = parse_args()
nvx, npx, n_cat = config.shapenet_32_64()
categories = labels['shapenetcore-v1']

edge = cv.imread(args.edge_path, 0).astype(np.float32) / 127.5 - 1
edge = edge.reshape((1, npx, npx, 1))

z = np.random.uniform(-1, 1, size=(1, args.nz))

label = np.zeros((1, n_cat))
label[:, categories[args.label]] = 1

model = sgan.Model(args.params_path)
x = model.generate(edge, z, label)
x = np.squeeze(x) > args.thresh
util.save_binvox(x, 'out/object.binvox')
