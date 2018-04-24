from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
from src.monitor_tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.monitor_siamese import SiameseNet
from src.monitor_driver import monitor_driver
import cv2 as cv
import serial

import torch.multiprocessing as mp
from torch.nn import DataParallel

RTSP = 'rtsp://192.168.31.105:554/user=admin&password=&channel=1&stream=1.sdp?'

def main():
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    siam = SiameseNet(env.root_pretrained, design.net).cuda()
    cap = cv.VideoCapture(RTSP)
    ser = serial.Serial('/dev/ttyUSB0', 2400)
    pos_x, pos_y, target_w, target_h = [176, 144, 60, 60] # center_x, center_y, w, h

    tracker(hp, design, cap, ser, pos_x, pos_y, target_w, target_h, final_score_sz, siam, True)




if __name__ == '__main__':
    sys.exit(main())

