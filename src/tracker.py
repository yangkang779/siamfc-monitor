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
import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores
from src.bbreg import *
from src.sample_generator import *
from src.data_prov import *

# read default parameters and override with custom ones
# def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame):
def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, siam, start_frame):
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames, 4))

    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    # min_z = hp.scale_min * z_sz
    # max_z = hp.scale_max * z_sz
    # min_x = hp.scale_min * x_sz
    # max_x = hp.scale_max * x_sz

    if True: # for replacing the sess.run()
        
        # save first frame position (from ground-truth)
        bboxes[0, :] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

        image_, templates_z_ = siam.get_template_z(pos_x, pos_y, z_sz, frame_name_list[0], design)
        new_templates_z_ = templates_z_

        # Train bbox regressor
        if design.bbreg is True:
            bbreg_examples = gen_samples(SampleGenerator('uniform', image_.size, 0.3, 1.5, 1.1),
                                         bboxes[0], 1000, [0.6, 1], [1, 2])
            bbreg_feats = forward_samples(siam, image_, bbreg_examples)  # bbreg_examples.shape = (477, 4) box's coord [x, y, w, h]
            bbreg = BBRegressor(image_.size)
            bbreg.train(bbreg_feats, bbreg_examples, bboxes[0])

        t_start = time.time()
        torch.set_num_threads(8)
        # Get an image from the queue
        for i in range(1, num_frames):
            tic = time.time()
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors

            image_, scores_ = siam.get_scores(pos_x, pos_y, scaled_search_area, templates_z_, frame_name_list[i], design, final_score_sz)
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            # update scaled sizes
            x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
            target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id,:,:]
            score_ = score_ - np.min(score_)
            score_ = score_/np.sum(score_)
            # apply displacement penalty
            score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[i, :] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

            if design.bbreg is True:
                bbreg_feats = forward_samples(siam, image_, np.expand_dims(bboxes[i], 0))
                bbreg_samples = bbreg.predict(bbreg_feats, np.expand_dims(bboxes[i], 0))
                bboxes[i, :] = bbreg_samples.mean(axis=0)

            # update the target representation with a rolling average
            # if hp.z_lr > 0:
            #     _, new_templates_z_ = siam.get_template_z(pos_x, pos_y, z_sz, image_, design)
            #
            #     templates_z_ = (1 - hp.z_lr) * templates_z_ + hp.z_lr * new_templates_z_
            
            # update template patch size
            z_sz = (1 - hp.scale_lr) * z_sz + hp.scale_lr * scaled_exemplar[new_scale_id]
            
            # if run.visualization:
            #     show_frame(image_, bboxes[i, :], 1)

            # print('Frame : {}, FPS : {:.2f} '.format(i + 1, 1 / (time.time() - tic)))

        t_elapsed = time.time() - t_start
        speed = num_frames / t_elapsed

    # plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

def forward_samples(model, image, samples):
    extractor = RegionExtractor(image, samples, 127, 16, 256)
    for i, regions in enumerate(extractor):
        regions = Variable(regions).cuda() #regions [128, 3, 107, 107]
        feat = model.mid_feature(regions) #feat.shape 128*4608
        feat = feat.view(feat.size(0), -1)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats
