import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from numpy import dtype
import sys

DEBUG = False

class VectorPoolLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception('Wrong No of bottom blobs (should be = 1)')
        
        layer_params = yaml.load(self.param_str_)
        #self._star_index = layer_params['start_index']
        self._pad = layer_params['pad']
        self._stride = layer_params['stride']
        self._kernel = layer_params['kernel']
        
        #computing output shape based on kernel information
        self._new_dim = (bottom[0].shape[1] - self._kernel + 2*self._pad) / self._stride + 1
        
        if DEBUG:
            print 'bottom[0].data.shape : ', bottom[0].data.shape
                
        top[0].reshape(bottom[0].data.shape[0], self._new_dim)
        
        
    def forward(self, bottom, top):
        
        #sanity check
        if DEBUG:
            print 'bottom[0].data.shape : ', bottom[0].data.shape
            #print 'bottom[0].data', bottom[0].data[...]
       
        data_vector =  bottom[0].data   
        # padding values
        pad_val = np.zeros((bottom[0].shape[0], self._pad), dtype = data_vector.dtype)
        padded_data = np.hstack((pad_val, data_vector, pad_val))
        if DEBUG:
            print ' Data Vector after padding ', padded_data[0:2]
            print ' Shape after padding ', padded_data.shape
            
        max = np.zeros((bottom[0].shape[0], self._new_dim))           #for max values
        self._maxArg = np.zeros((bottom[0].shape[0], self._new_dim), dtype=int)  #for index
        
        if DEBUG:
            print 'Max shape: ', max.shape, ' Argmax.shape: ', self._maxArg.shape

        for i in xrange(0, padded_data.shape[1]-1, self._stride):
            ind = range(i,i+self._stride)            
            max[:,i/self._stride] = np.max(padded_data[:,ind], axis=1)
            self._maxArg[:,i/self._stride] = np.argmax(padded_data[:,ind], axis=1) + np.full(padded_data.shape[0], i-self._pad)
            
        #resetting index for background class to 0
        self._maxArg[:,0] = 0.0
            
        if DEBUG:
            print ' Max pool o/p: ', max[0:2], ' Argmax : ', self._maxArg[0:2]
            #sys.exit()
            
            
        #top[0].data[0] = 0          # Since single batch
        top[0].reshape(*max.shape)
        top[0].data[...] = max        # Max values
        
        if DEBUG:
            print 'Top Data : ', top[0].data[0:2]
            print 'Top Data shape: ', top[0].data.shape
        
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        
        if DEBUG:
            print ' In backward prop top diff : ', top[0].diff[0:2]
            print ' Argmax: ', self._maxArg[0:2]
            print 'bottom[0].diff: ', bottom[0].diff.shape
            
        
        top_grad = top[0].diff[...]
        bottom_grad = np.zeros(bottom[0].diff.shape)
        
        for i in range(top_grad.shape[0]):
            for j in range(top_grad.shape[1]):
                bottom_grad[i][self._maxArg[i][j]] = top_grad[i][j]
        
        if DEBUG:
            print 'Gradient Val : ', bottom_grad[0:2]
            
        bottom[0].diff[...] = bottom_grad

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass