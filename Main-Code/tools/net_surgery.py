import _init_paths
import caffe
import numpy as np

# train.prototxt of the architectures
#zf_prototxt = '../output/zf_FRCNN.prototxt' 
# class_only_prototxt = './output/class_only.prototxt'
# baseline_prototxt = './output/baseline.prototxt'
arch_3_prototxt = '../models/Arch_3_bb_11_fc8_init/train.prototxt'

# trained models file
#zf_model = '../data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel'
# class_only_file = './output/class_only_iter_100000.caffemodel'
# baseline_file = './output/baseline_iter_100000.caffemodel'
arch_3_model = '../output/Arch_3_bb_11_fc8_init/train/arch_3_bb_11_fc8_init_iter_0.caffemodel'

# class_net = caffe.Net(class_only_prototxt, class_only_file, caffe.TRAIN)
# baseline_net = caffe.Net(baseline_prototxt, baseline_file, caffe.TRAIN)
arch_3 = caffe.Net(arch_3_prototxt, arch_3_model, caffe.TRAIN)
#zf_net = caffe.Net(zf_prototxt, zf_model, caffe.TRAIN)

'''
print '------------------------------------------------------------------------'
print '------------------------- ZF FRCNN  Network Architecture---------------------------'
print '------------------------------------------------------------------------'
for layername, layerparam in zf_net.params.items():
    print '  Layer Name : {0:>7}, Weight Dims :{1:12} '.format(layername, layerparam[0].data.shape)
print '------------------------------------------------------------------------'
'''

print '------------------------------------------------------------------------'
print '------------------------- Arch_3  Network Architecture---------------------------'
print '------------------------------------------------------------------------'
for layername, layerparam in arch_3.params.items():
    print '  Layer Name : {0:>7}, Weight Dims :{1:12} '.format(layername, layerparam[0].data.shape)
print '------------------------------------------------------------------------'

'''
# conv1
arch_3.params['conv1'][0].data[...] = zf_net.params['conv1'][0].data[...]
arch_3.params['conv1'][1].data[...] = zf_net.params['conv1'][1].data[...] 

# conv2
arch_3.params['conv2'][0].data[...] = zf_net.params['conv2'][0].data[...]
arch_3.params['conv2'][1].data[...] = zf_net.params['conv2'][1].data[...]

# conv3
arch_3.params['conv3'][0].data[...] = zf_net.params['conv3'][0].data[...]
arch_3.params['conv3'][1].data[...] = zf_net.params['conv3'][1].data[...]

# conv4
arch_3.params['conv4'][0].data[...] = zf_net.params['conv4'][0].data[...]
arch_3.params['conv4'][1].data[...] = zf_net.params['conv4'][1].data[...]

# conv5
arch_3.params['conv5'][0].data[...] = zf_net.params['conv5'][0].data[...]
arch_3.params['conv5'][1].data[...] = zf_net.params['conv5'][1].data[...]
'''

'''
# rpn_conv/3x3
arch_3.params['rpn_conv/3x3'][0].data[...] = baseline_net.params['rpn_conv/3x3'][0].data[...]
arch_3.params['rpn_conv/3x3'][1].data[...] = baseline_net.params['rpn_conv/3x3'][1].data[...]

# rpn_cls_score
arch_3.params['rpn_cls_score'][0].data[...] = baseline_net.params['rpn_cls_score'][0].data[...]
arch_3.params['rpn_cls_score'][1].data[...] = baseline_net.params['rpn_cls_score'][1].data[...]

# rpn_bbox_pred
arch_3.params['rpn_bbox_pred'][0].data[...] = baseline_net.params['rpn_bbox_pred'][0].data[...]
arch_3.params['rpn_bbox_pred'][1].data[...] = baseline_net.params['rpn_bbox_pred'][1].data[...]

'''
'''
# Intializing class only fc fc7 of network
# fc6
arch_3.params['fc6'][0].data[...] = zf_net.params['fc6'][0].data[...]
arch_3.params['fc6'][1].data[...] = zf_net.params['fc6'][1].data[...]

# fc7
arch_3.params['fc7'][0].data[...] = zf_net.params['fc7'][0].data[...]
arch_3.params['fc7'][1].data[...] = zf_net.params['fc7'][1].data[...]
'''

# fc8
arch_3.params['fc8'][0].data[...] = arch_3.params['fc7'][0].data[...]
arch_3.params['fc8'][1].data[...] = arch_3.params['fc7'][1].data[...]


arch_3.save('./output/Arch_3_bb_11_fc8_init/train/arch_3_bb_11_fc8_init_0.caffemodel')
