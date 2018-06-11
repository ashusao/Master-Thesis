# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import sys

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        st_scores = blobs_out['state_prob']      #Ashu output blob state score

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        #box_deltas = blobs_out['bbox_pred']
        box_deltas = net.blobs['bbox_pred'].data    #Ashu for arch 4 only
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, st_scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    
     # all detections are collected into:                            Ashu for states
    #    all_boxes[state][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes_state = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_states)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    
    #fractions of states in class statics
    f_closedLaptop = 396.0/(396.0+3014.0)
    f_openLaptop = 3014.0/(396.0+3014.0)
    f_closedScissor = 745.0/(745.0+315.0)
    f_openScissor = 315.0/(745.0+315.0)
    f_closedSuitcase = 3917.0/(3917.0+301.0)
    f_openSuitcase = 301.0/(3917.0+301.0)
    f_closedToilet = 1359.0/(1359.0+1483.0)
    f_openToilet = 1483.0/(1359.0+1483.0)
    f_closedUmbrella = 647.0/(647.0+7167.0)
    f_openUmbrella = 7167.0/(647.0+7167.0)
    

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, st_scores, boxes = im_detect(net, im, box_proposals)     #Ashu state score
        _t['im_detect'].toc()
        
        ''' Weights and biases
        params = ['fc6', 'fc7', 'cls_score']
        # fc_params = {name: (weights, biases)}
        fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

        for fc in params:
            print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
            print fc_params[fc]
        
        sys.exit()'''
        
        ''' Neuron activations
        '''
        '''
        layer_outputs = ['fc6', 'fc7','fc8' , 'cls_score', 'bbox_pred']
        for ly in layer_outputs:
            res = net.blobs[ly].data
            print ly + ' Shape: ' + str(res.shape)
            print 'Value : ' 
            print res
            print ' Mean : ' + str(np.mean(res, axis=1))
        
        sys.exit()'''
        
        st_scores_new = np.zeros(st_scores.shape, dtype=np.float)
        #Adding background score
        st_scores_new[:,0] = scores[:,0]
        st_scores_new[:,1] = scores[:,1]*f_openScissor
        st_scores_new[:,2] = scores[:,1]*f_closedScissor
        st_scores_new[:,3] = scores[:,2]*f_openUmbrella
        st_scores_new[:,4] = scores[:,2]*f_closedUmbrella
        st_scores_new[:,5] = scores[:,3]*f_openToilet
        st_scores_new[:,6] = scores[:,3]*f_closedToilet
        st_scores_new[:,7] = scores[:,4]*f_openSuitcase
        st_scores_new[:,8] = scores[:,4]*f_closedSuitcase
        st_scores_new[:,9] = scores[:,5]*f_openLaptop
        st_scores_new[:,10] = scores[:,5]*f_closedLaptop
        
        '''
        print 'scores.shape ', scores.shape
        print 'st_scores.shape ', st_scores.shape
        
        print 'st_scores val : ', st_scores[0]
        print 'cls_scores val : ', scores[0]
        print 'st_scores new val : ', st_scores_new[0]
        
        sys.exit()'''

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            
            cls_boxes_1 = boxes[inds, (2*j - 1)*4 : 2*j*4]                        #Ashu adding another box cooresponding to 2nd state
            cls_boxes_2 = boxes[inds, 2*j*4 : (2*j + 1)*4]
           # print 'class boxes ', cls_boxes
            cls_dets_1 = np.hstack((cls_boxes_1, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets_2 = np.hstack((cls_boxes_2, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets = np.vstack((cls_dets_1, cls_dets_2))
            #print 'cls_dets.shape : ', cls_dets.shape
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets
            #print 'inds : ', inds, ' keep :', keep
            #print 'cls_dets1 : ', cls_dets_1
            #print 'cls_dets2 : ', cls_dets_2
            #print ' cls_dets : ', cls_dets
            #print 'len cls_dets 1 : ', len(cls_dets_1)
            #print 'len cls_dets : ', len(cls_dets)
            #print 'j : ', j, ' i: ', i, ' boxes : ',all_boxes[j][i]
            
        # skip j = 0, because it's the background state                    Ashu loop for state score
        for j in xrange(1, imdb.num_states):
            inds = np.where(st_scores[:, j] > thresh)[0]
            state_scores = st_scores[inds, j]
            
            #inds = np.where(st_scores_new[:, j] > thresh)[0]    #modified for cond prob
            #state_scores = st_scores_new[inds, j]               #modified for cond prob
            
            state_boxes = boxes[inds, j*4:(j+1)*4]              #Ashu .. since for every class 2 states.. have same bbox cord
            #print 'State boxes ', state_boxes
            state_dets = np.hstack((state_boxes, state_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            #print 'state_dets.shape : ', state_dets.shape
            keep = nms(state_dets, cfg.TEST.NMS)
            state_dets = state_dets[keep, :]
            if vis:
                vis_detections(im, imdb.states[j], state_dets)
            all_boxes_state[j][i] = state_dets
            #print 'inds : ', inds, ' keep :', keep
            #print 'j : ', j, ' i: ', i, ' boxes : ',all_boxes_state[j][i]
                
        
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    
        # Limit to max_per_image detections *over all states*            Ashu loop for limiting state detections
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes_state[j][i][:, -1]
                                      for j in xrange(1, imdb.num_states)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_states):
                    keep = np.where(all_boxes_state[j][i][:, -1] >= image_thresh)[0]
                    all_boxes_state[j][i] = all_boxes_state[j][i][keep, :]
                    
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
        #sys.exit()
              
    #print 'All boxes : ', all_boxes

    det_file_cls = os.path.join(output_dir, 'detections_cls.pkl')
    with open(det_file_cls, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
        
    '''
    tmp = np.array(all_boxes_state)
    print 'all_boxes_state.shape ', tmp.shape    
    print 'all_boxes_state value : '
    print all_boxes_state'''
        
    det_file_state = os.path.join(output_dir, 'detections_state.pkl')
    with open(det_file_state, 'wb') as f:
        cPickle.dump(all_boxes_state, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, all_boxes_state, output_dir)
