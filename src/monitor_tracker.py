import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
import torch
from torch.autograd import Variable
from src.visualization import show_frame, show_crops, show_scores
from src.bbreg import *
from src.sample_generator import *
from src.data_prov import *
from src.monitor_driver import monitor_driver
import cv2 as cv

distance = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

def tracker(hp, design, cap, ser, pos_x, pos_y, target_w, target_h, final_score_sz, siam, display):

    scale_factors = hp.scale_step ** np.linspace(-np.ceil(hp.scale_num / 2), np.ceil(hp.scale_num / 2), hp.scale_num)
    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # save first frame position (from ground-truth)
    init_box = np.array([pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h]) # top-left
    bbox_prev = init_box

    ret, init_frame = cap.read()
    frame_num = 0
    if ret is True:
        print('initialize first frame ...')
        image = Image.fromarray(init_frame).convert('RGB')
        image_, templates_z_ = siam.get_template_z(pos_x, pos_y, z_sz, image, design)
        new_templates_z_ = templates_z_
        # Train bbox regressor
        if design.bbreg is True:
            print('training boundingbox regressor ...')
            bbreg_examples = gen_samples(SampleGenerator('uniform', image_.size, 0.3, 1.5, 1.1),
                                         init_box, 256, [0.6, 1], [1, 2])
            bbreg_feats = forward_samples(siam, image_, bbreg_examples)  # bbreg_examples.shape = (477, 4) box's coord [x, y, w, h]
            bbreg = BBRegressor(image_.size)
            bbreg.train(bbreg_feats, bbreg_examples, init_box)

    torch.set_num_threads(8)
    print('start tracking !')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True and frame_num % 3 == 0:
            tic = time.time()
            image = Image.fromarray(frame).convert('RGB')
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors
            image_, scores_ = siam.get_scores(pos_x, pos_y, scaled_search_area, templates_z_, image, design, final_score_sz)
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0, :, :] = hp.scale_penalty * scores_[0, :, :]
            scores_[2, :, :] = hp.scale_penalty * scores_[2, :, :]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
            # update scaled sizes
            x_sz = (1 - hp.scale_lr) * x_sz + hp.scale_lr * scaled_search_area[new_scale_id]
            target_w = (1 - hp.scale_lr) * target_w + hp.scale_lr * scaled_target_w[new_scale_id]
            target_h = (1 - hp.scale_lr) * target_h + hp.scale_lr * scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id, :, :]
            score_ = score_ - np.min(score_)
            score_ = score_ / np.sum(score_)
            # apply displacement penalty
            score_ = (1 - hp.window_influence) * score_ + hp.window_influence * penalty
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride,
                                                   design.search_sz, hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            frame_box = [pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h]

            if design.bbreg is True:
                bbreg_feats = forward_samples(siam, image_, np.expand_dims(frame_box, 0))
                bbreg_samples = bbreg.predict(bbreg_feats, np.expand_dims(frame_box, 0))
                frame_box = bbreg_samples.mean(axis=0)

            # update the target representation with a rolling average
            if hp.z_lr > 0:
                _, new_templates_z_ = siam.get_template_z(pos_x, pos_y, z_sz, image_, design)

                templates_z_ = (1 - hp.z_lr) * templates_z_ + hp.z_lr * new_templates_z_
            # update template patch size

            z_sz = (1 - hp.scale_lr) * z_sz + hp.scale_lr * scaled_exemplar[new_scale_id]

            euc_dis = distance([bbox_prev[0] + bbox_prev[2] / 2, bbox_prev[1] + bbox_prev[3] / 2],
                               [frame_box[0] + frame_box[2] / 2, frame_box[1] + frame_box[3] / 2])

            if euc_dis >= 20:
                ser_dirver = monitor_driver(image.size, frame_box, 0.3)
                ser.write(ser_dirver.tracking())
                bbox_prev = frame_box

                print('boundingbox : {}  fps:{:.2f} '.format(frame_box, 1 / (time.time() - tic)))

            if display:
                display_frame = np.asarray(image)
                cv.rectangle(display_frame, (int(frame_box[0]), int(frame_box[1])),
                             (int(frame_box[0] + frame_box[2]), int(frame_box[1] + frame_box[2])), (0, 0, 255), 2)
                cv.namedWindow('display', cv.WINDOW_NORMAL)
                cv.imshow('display', display_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_num += 1

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

def forward_samples(model, image, samples):
    extractor = RegionExtractor(image, samples, 107, 16, 256)
    for i, regions in enumerate(extractor):
        regions = Variable(regions).cuda()  # regions [128, 3, 107, 107]
        feat = model.bbreg_feature(regions)  # feat.shape 128*4608
        feat = feat.view(feat.size(0), -1)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats
