# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import sys
#from basketball_eval import basketball_eval
from stateDetection_eval import stateDetection_eval
from fast_rcnn.config import cfg

class stateDetection(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set     # Ashu train2014 / val2014
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'scissors', 
                         'umbrella', 
                         'toilet', 
                         'suitcase', 
                         'laptop')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))  # Ashu indices for each class
        self._states = ('__background__', # always index 0
                         'openScissor', 'closedScissor',
                         'openUmbrella', 'closedUmbrella',
                         'openToilet', 'closedToilet',
                         'openSuitcase', 'closedSuitcase',
                         'openLaptop', 'closedLaptop')
        self._state_to_ind = dict(zip(self.states, xrange(self.num_states)))
        self._image_ext = ['.jpg','.png','.JPEG']
        self._image_index = self._load_image_set_index()    # Ashu contains the list of image Ids in train.txt
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : False,       # Ashu to keep detections
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_stateDetection_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_stateDetection_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the stateDetection
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        
        #since difficult = 0 for all images excluding this filtering
        
       # if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
         #   non_diff_objs = [
          #      obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
         #   objs = non_diff_objs
            
            
            
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_states = np.zeros((num_objs), dtype=np.int32)                #Ashu 
        #overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        overlaps = np.zeros((num_objs, self.num_states), dtype=np.float32) #Ashu modified for bbox 11
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) 
            y1 = float(bbox.find('ymin').text) 
            x2 = float(bbox.find('xmax').text) 
            y2 = float(bbox.find('ymax').text) 
#            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            cls = self._class_to_ind[obj.find('name').text.split("_")[0].strip()]
            st = self._state_to_ind[obj.find('name').text.split("_")[1].strip()]    #Ashu Extract state info
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            gt_states[ix] = st                                          #Ashu assign gt_state
            overlaps[ix, st] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,            #Ashu [x1, y1, x2, y2]
                'gt_classes': gt_classes,   #Ashu index of class
                'gt_states': gt_states,   #Ashu index of class
                'gt_overlaps' : overlaps,   #Ashu Sparse Matrix num_obj x num_state
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_stateDetection_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        if not (os.path.exists(self._devkit_path + "/results/" + cfg.EXP_DIR)):
            os.makedirs(self._devkit_path + "/results/" + cfg.EXP_DIR)

        path = os.path.join(
            self._devkit_path,
            'results', 
            cfg.EXP_DIR,
            filename)
        
        return path

    def _write_stateDetection_results_file(self, all_boxes, all_boxes_state):

        #Writing Class detectoins
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} stateDetection results file'.format(cls)
            filename = self._get_stateDetection_results_file_template().format(cls)
            print filename
            with open(filename, 'wt') as f:
                #f.write('Image_Name Confidence x1 y1 x2 y2\n')      #Generated file format
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                        # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                        dets[k, 0] + 1, dets[k, 1] + 1,
                                        dets[k, 2] + 1, dets[k, 3] + 1))
                        
        #Writing state detectoins
        for st_ind, st in enumerate(self.states):
            if st == '__background__':
                continue
            print 'Writing {} stateDetection results file'.format(st)
            filename = self._get_stateDetection_results_file_template().format(st)
            print filename
            with open(filename, 'wt') as f:
                #f.write('Image_Name Confidence x1 y1 x2 y2\n')      #Generated file format
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes_state[st_ind][im_ind]
                    if dets == []:
                        continue
                        # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                        dets[k, 0] + 1, dets[k, 1] + 1,
                                        dets[k, 2] + 1, dets[k, 3] + 1))
            
    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache', cfg.EXP_DIR)
        
        #Calculating Result for class only
        aps_cls = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_stateDetection_results_file_template().format(cls)
            rec, prec, ap = stateDetection_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps_cls += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Class Mean AP = {:.4f}'.format(np.mean(aps_cls)))
        print('~~~~~~~~')
        print('Results Class:')
        for ap in aps_cls:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps_cls)))
        print('~~~~~~~~')
        print('')
        print('########################################')
        
        #Calculating Result for state only
        aps_st = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, st in enumerate(self._states):
            if st == '__background__':
                continue
            filename = self._get_stateDetection_results_file_template().format(st)
            rec, prec, ap = stateDetection_eval(
                filename, annopath, imagesetfile, st, cachedir, ovthresh=0.5)
            aps_st += [ap]
            print('AP for {} = {:.4f}'.format(st, ap))
            with open(os.path.join(output_dir, st + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Sate Mean AP = {:.4f}'.format(np.mean(aps_st)))
        print('~~~~~~~~')
        print('Results State:')
        for ap in aps_st:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps_st)))
        print('~~~~~~~~')
        print('')
        
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, all_boxes_state, output_dir):
        self._write_stateDetection_results_file(all_boxes, all_boxes_state)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_stateDetection_results_file_template().format(cls)
                os.remove(filename)

if __name__ == '__main__':
    from datasets.stateDetection import stateDetection
    d = stateDetection('trainval')
    res = d.roidb
    from IPython import embed; embed()
