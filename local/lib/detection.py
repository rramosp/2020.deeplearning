import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import savemat
from skimage.segmentation import felzenszwalb
import time
from sklearn.preprocessing import normalize
from skimage.filters import gaussian
import shutil
import os
from skimage import color


def convert_colorspace(img,colorspace_list):
    """ Converts RGB image to the formats in colorspace_list
    Paramters
    ---------
    img : Input Image
    colorspace_list : string list of colorspaces to be converted to. This param can also be a string
    Possible strings are ['RGB', 'I', 'LAB', 'rgI', 'HSV', 'rgb', 'C', 'H']
    Returns
    --------
    out_arr : list of images in various colorspaces. Shape: (|colorspace_list|, )
    """



    colorspace = np.atleast_1d(colorspace_list)

    out_arr = [[]]*len(colorspace)

    for i,colorspace in enumerate(colorspace_list):
        
        if colorspace == 'RGB':
            if img.max()>1: out_arr[i] = img/255.0
            else: out_arr[i] = img

        elif colorspace == 'I':
            out_arr[i] = color.rgb2gray(img)

        elif colorspace == 'LAB':
            out_arr[i] = color.rgb2lab(img)
            out_arr[i][:,:,0] = out_arr[i][:,:,0]/100.0
            out_arr[i][:,:,1] = (out_arr[i][:,:,1]+127)/255.0    
            out_arr[i][:,:,2] = (out_arr[i][:,:,1]+127)/255.0

        elif colorspace == 'rgI':
            out_arr[i] = np.zeros(img.shape)
            out_arr[i][:,:,0:2] = img[:,:,0:2]
            out_arr[i][:,:,2] = color.rgb2gray(img)

        elif colorspace == 'HSV':
            out_arr[i] = color.rgb2hsv(img)

        elif colorspace == 'rgb':
            out_arr[i] = rgb2rgb_norm(img)

        elif colorspace == 'C':
            out_arr[i] = rgb2C(img)

        elif colorspace == 'H:':
            out_arr[i] == color.rgb2hsv(img)[:,:,0]

        else:
            print('Not Implemented. Error')
            return None

    return out_arr

def rgb2C(img):
    """ Converts RGB to Opponent color space
    Paramters
    ---------
    img : Input Image
    Returns
    --------
    out_arr : Opponent colorspace image
    Refer to https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Opponent.pdf for more details
    """
    
    out_arr = np.zeros(img.shape)
    out_arr[:,:,0] = color.rgb2lab(img)[:,:,0]
    out_arr[:,:,1] = img[:,:,1] - img[:,:,0]
    out_arr[:,:,2] = img[:,:,2] - (img[:,:,1] + img[:,:,0])
    return out_arr


def rgb2rgb_norm(img):
    """ Converts RGB to normalised RGB color space
    Paramters
    ---------
    img : Input Image
    Returns
    --------
    out_arr : normalised RGB colorspace image
    """    
    temp_I = I / 255.0
    norm = np.sqrt(temp_I[:, :, 0] ** 2 + temp_I[:, :, 1] ** 2 + temp_I[:, :, 2] ** 2)
    out_arr = np.zeros(img.shape)
    out_arr[:,:,0] = (temp_I[:, :, 0] / norm * 255).astype(numpy.uint8)
    out_arr[:,:,0] = (temp_I[:, :, 1] / norm * 255).astype(numpy.uint8)
    out_arr[:,:,0] = (temp_I[:, :, 2] / norm * 255).astype(numpy.uint8)
    return out_arr



class blob:
    """ 
    Blob : An image region or segment
    Parameters
    ----------
    
    blob_idx : Blob Index
    
    blob_size : no of pixels that constitute the blob_size
    bbox : A tight bounding box that encloses the blob_size
    neighbours : blob_idx of the neighbouring blobs
    color_hist : Color histogram of the blob
    texture_hist : Texture histogram of the blob_size
    """


    def __init__(self,idx,blob_size=None,bbox=None):

        self.blob_idx = idx

        if not blob_size is None:
            self.blob_size = blob_size

        if not bbox is None:
            self.bbox = bbox

        self.neighbours = set()

        self.color_hist = []

        self.texture_hist = []

def get_texture_hist(img,segment_mask,n_orientation = 8, n_bins = 10):
    ''' 
    Computes texture histograms for all the blobs
    parameters
    ----------
    img : Input Image
    segment_ mask :  Integer mask indicating segment labels of an image
    returns
    -------
    
    hist : texture histogram of the blobs. Shape: [ n_segments , n_bins*n_orientations*n_color_channels ]
    '''
    filt_img = gaussian(img, sigma = 1.0, multichannel = True).astype(np.float32)
    op = np.array([[-1.0, 0.0, 1.0]])
    grad_x = np.array([scipy.ndimage.filters.convolve(filt_img[:,:,i], op) for i in range(img.shape[-1])])
    grad_y = np.array([scipy.ndimage.filters.convolve(filt_img[:,:,i], op.T) for i in range(img.shape[-1])])
    _theta = np.arctan2(grad_y, grad_y)
    theta = np.zeros(img.shape)
    for i in range(img.shape[-1]):theta[:,:,i] = _theta[i]
    n_segments = len(set(segment_mask.flatten()))
    labels = range(n_segments + 1)    
    bins_orientation = np.linspace(-np.pi, np.pi, n_orientation + 1)
    bins_intensity = np.linspace(0.0, 1.0, n_bins + 1)
    bins = [labels, bins_orientation, bins_intensity]
    _temp = [ np.vstack([segment_mask.flatten(), theta[:,:,i].flatten(), filt_img[:,:,i].flatten()]).T for i in range(img.shape[-1])]
    hist = np.hstack([ np.histogramdd(_temp[i], bins = bins)[0] for i in range(img.shape[-1]) ])
    hist = np.reshape(hist,(n_segments,n_orientation*n_bins*img.shape[-1]))
    hist = normalize(hist,norm='l1',axis=1)
    return hist

def get_blob_neighbours(blob_array,segment_mask):

    """ Set the neighbour attribute of blob class
    Parameters
    ----------
    blob_array : Array of blobs
    segment_mask : Integer mask indicating segment labels of an image
    Returns
    -------
    neighbour_set : Set of neighbours ordered as tuples
    """


    idx_neigh = np.where(segment_mask[:,:-1]!=segment_mask[:,1:])
    x_neigh = np.vstack((segment_mask[:,:-1][idx_neigh],segment_mask[:,1:][idx_neigh])).T
    x_neigh = np.sort(x_neigh,axis=1)
    x_neigh = set([ tuple(_x) for _x in x_neigh])

    idy_neigh = np.where(segment_mask[:-1,:]!=segment_mask[1:,:])
    y_neigh = np.vstack((segment_mask[:-1,:][idy_neigh],segment_mask[1:,:][idy_neigh])).T
    y_neigh = np.sort(y_neigh,axis=1)
    y_neigh = set([ tuple(_y) for _y in x_neigh])

    neighbour_set = x_neigh.union(y_neigh)

    for _loc in neighbour_set:
        blob_array[_loc[0]].neighbours.add(_loc[1])
        blob_array[_loc[1]].neighbours.add(_loc[0])
    return neighbour_set
                
def merge_blobs(blob_array,blob_1,blob_2,t):

    """ Merges two blobs and updates the blob_dict
    Parameters 
    -----------
    blob_dict : Dictionary of blobs with their id as key
    blob_id1, blob_id2 : The ids of the blobs to be merged
    t : The id to be assigned to the new blob
    """

    blob_t = blob(t)

    blob_t.blob_size = blob_1.blob_size + blob_2.blob_size

    blob_t.neighbours = blob_1.neighbours.union(blob_2.neighbours)
    
    for idx in blob_1.neighbours:
        if idx ==t: continue
        blob_array[idx].neighbours.remove(blob_1.blob_idx)
        blob_array[idx].neighbours.add(t)    

    for idx in blob_2.neighbours:
        if idx==t: continue
        blob_array[idx].neighbours.remove(blob_2.blob_idx)
        blob_array[idx].neighbours.add(t)    

    blob_t.neighbours.remove(blob_1.blob_idx)
    blob_t.neighbours.remove(blob_2.blob_idx)

        

    blob_t.bbox = np.empty(4)
    blob_t.bbox[0] = min(blob_1.bbox[0], blob_2.bbox[0])
    blob_t.bbox[1] = min(blob_1.bbox[1], blob_2.bbox[1])
    blob_t.bbox[2] = max(blob_1.bbox[2], blob_2.bbox[2])
    blob_t.bbox[3] = max(blob_1.bbox[3], blob_2.bbox[3])
    
    # Merge color_hist
    blob_t.color_hist = (blob_1.color_hist*blob_1.blob_size + blob_2.color_hist*blob_2.blob_size)/blob_t.blob_size

    # Merge texture_hist
    blob_t.texture_hist = (blob_1.texture_hist*blob_1.blob_size + blob_2.texture_hist*blob_2.blob_size)/blob_t.blob_size

    return blob_t

    

def get_color_hist(img,segment_mask,n_bins=25):
    ''' 
    Computes color histograms for all the blobs
    parameters
    ----------
    img : Input Image
    segment_ mask :  Integer mask indicating segment labels of an image
    returns
    -------
    
    hist : color_histogram of the blobs. Shape: [ n_segments , n_bins*n_color_channels ]
    '''
    if img.max()>1:    _img = img/255.0
    else: _img = img
    n_segments = len(set(segment_mask.flatten()))
    bins = np.linspace(0.0,1.0,n_bins+1)
    labels = range(n_segments + 1)
    bins = [labels, bins]
    hist = np.hstack([ np.histogram2d(segment_mask.flatten(), _img[:, :, i].flatten(), bins=bins)[0] for i in range(img.shape[-1]) ])
    hist = normalize(hist,norm='l1',axis=1)
    
    return hist

def color_hist_sim():
    return lambda blob_1, blob_2 : np.minimum(blob_1.color_hist,blob_2.color_hist).sum()

def texture_hist_sim():
    return lambda blob_1, blob_2 : np.minimum(blob_1.texture_hist,blob_2.texture_hist).sum()

def size_sim(shape):
    return lambda blob_1, blob_2 : 1 - (blob_1.blob_size + blob_2.blob_size)*1.0/(shape[0]*shape[1])

def fill_sim(shape):
    return lambda blob_1, blob_2 : 1 - compute_fill(blob_1, blob_2, shape)


def compute_sim(blob_1,blob_2,sim_feats):
    ''' Helper function to compute similarity '''
    similarity = 0
    for _sim_feat in sim_feats:
        similarity += _sim_feat(blob_1,blob_2)
    return similarity

def compute_fill(blob_1,blob_2,shape):

    BBox = [[]]*4
    BBox[0] = min(blob_1.bbox[0],blob_1.bbox[0])
    BBox[1] = min(blob_1.bbox[1],blob_1.bbox[1])
    BBox[2] = max(blob_1.bbox[2],blob_1.bbox[2])
    BBox[3] = max(blob_1.bbox[3],blob_1.bbox[3])

    BBox_size = abs(BBox[0]-BBox[2])*abs(BBox[1]-BBox[3])
    fill = (BBox_size - blob_1.blob_size - blob_2.blob_size)*1.0/(shape[0]*shape[1])
    return fill

def _ssearch(img,segment_mask,sim_feats=None):

    '''
    Performs selective_search on the given image
    parameters
    ----------
        
    img : Input image
    
    segment_ mask :  Integer mask indicating segment labels of an image
    
    sim_feats : list of sim_features to be used
    Default(None) : [ color_hist_sim(),texture_hist_sim(),size_sim(img),fill_sim(img) ]
    
    returns
    --------
    
    blob_array : Array of blobs computed during the hierarchial process
    ''' 
    a = time.time()    
    h = img.shape[0]
    w = img.shape[1]
    n_segments = len(set(segment_mask.flatten()))    
    blob_sizes = np.bincount(segment_mask.flatten())
    color_hists = get_color_hist(img,segment_mask,n_bins=25)
    texture_hists = get_texture_hist(img,segment_mask,n_orientation = 8, n_bins = 10)
    blob_array = []
    for i in range(n_segments):        
        blob_array.append(blob(i))
        _loc = np.argwhere(segment_mask==i)
        bbox = np.empty(4)
        bbox[0] = _loc[:,0].min()
        bbox[1] = _loc[:,1].min()
        bbox[2] = _loc[:,0].max()
        bbox[3] = _loc[:,1].max()
        blob_array[i].blob_size = blob_sizes[i]
        blob_array[i].bbox = bbox
        blob_array[i].color_hist = color_hists[i]
        blob_array[i].texture_hist = texture_hists[i]
    if sim_feats is None:
        sim_feats = [ sf.color_hist_sim(), sf.texture_hist_sim(), sf.size_sim(img), sf.fill_sim(img) ]
    neighbour_list = np.asarray(list(get_blob_neighbours(blob_array,segment_mask)))
    sim_list = np.vstack((neighbour_list.T,np.array([ compute_sim(blob_array[_idx[0]],blob_array[_idx[1]],sim_feats) for _idx in neighbour_list ]))).T

    while len(sim_list):
        
        # Get Max sim    
            
        sort_idx = np.argsort(sim_list[:,2])
        sim_list = sim_list[sort_idx]    
        blob_1 = blob_array[int(sim_list[-1][0])]
        blob_2 = blob_array[int(sim_list[-1][1])]        
        sim_list = sim_list[:-1]
        
        
        # Merge blobs
        
        t = len(blob_array)    
        blob_t = merge_blobs(blob_array,blob_1,blob_2,t)
        blob_array.append(blob_t)
            
        
        if len(sim_list)==0:
            break

        # Remove i,j from neighbour_list

        sim_list = sim_list[(sim_list[:,0]!=blob_1.blob_idx) & (sim_list[:,1]!=blob_1.blob_idx)]
        sim_list = sim_list[(sim_list[:,0]!=blob_2.blob_idx) & (sim_list[:,1]!=blob_2.blob_idx)]
        new_sim_list = np.array([[i,t,compute_sim(blob_array[i],blob_array[t],sim_feats)] for i in blob_t.neighbours])
        if len(new_sim_list):        
            sim_list = np.vstack((sim_list,new_sim_list))


    print('.'),    
    return blob_array

def load_segment_mask(filename):
    '''Loads Segment mask pre-computed and stored at filename'''
    return scipy.io.loadmat(filename)['blobIndIm'] - 1

def remove_duplicate(blob_array,priority):
    ''' Removes Duplicate Boxes 
    parameters
    -----------
    
    blob_array : array of blob_arrays for various strategies
    
    priority : array of priority arrays associated with blobs in blob_array
    
    returns
    -------
    bboxes : unique set of boxes with priorities. Shape [n_blobs,4]
    Note: box is of the form [xmin,xmax,ymin,ymax] ie img[xmin:xmax,ymin:ymax] denoted the selected region
    '''
    
    _boxes = [ tuple(_blob.bbox) for __blob in blob_array for _blob in __blob]
    _priority = np.asarray([ p for _p in priority for p in _p])
    _unq_boxes = set(_boxes)
    _boxes = np.asarray(_boxes)
    
    sort_idx = np.argsort(_priority)
    _priority = _priority[sort_idx]
    _boxes = _boxes[sort_idx]
    bboxes = []
    for _box,_p in zip(_boxes,_priority):
        if tuple(_box) in _unq_boxes:
            bboxes.append(np.append(_p,_box))
            _unq_boxes.remove(tuple(_box))    
    return bboxes

def selective_search(img, ks=[50,100]):

    blob_array = []
    priority = []
    seg_dir = '/tmp/'
    color_space_list = ['HSV','LAB']
    sim_feats_list = [[ color_hist_sim(), texture_hist_sim(), size_sim(img.shape), fill_sim(img.shape) ],[ texture_hist_sim(), size_sim(img.shape), fill_sim(img.shape) ]]

    cc = convert_colorspace(img,color_space_list)

    for i in range(len(color_space_list)):
        for j in range(len(ks)):
            for k in range(len(sim_feats_list)):
                _img = cc[i]
                _file = "%s%s/%d/%s.mat"%(seg_dir,color_space_list[i].upper(),ks[j],"tmp_image")
                fdir = "%s%s/%d"%(seg_dir,color_space_list[i].upper(),ks[j])
                if os.path.isdir(fdir):
                    shutil.rmtree(fdir)
                os.makedirs(fdir)
                if not os.path.exists(_file):
                    segment_mask = felzenszwalb(_img,scale=ks[j],sigma=0.8,min_size=ks[j])
                    _temp_dict = dict()
                    _temp_dict['blobIndIm'] = segment_mask + 1
                    savemat(_file,_temp_dict)
                _blob_array = _ssearch(_img,load_segment_mask(_file),sim_feats = sim_feats_list[k])
                blob_array.append(_blob_array)
                priority.append( np.arange(len(_blob_array),0,-1).clip(0,(len(_blob_array)+1)/2))

    bboxes = remove_duplicate(blob_array,priority)
    bboxes = np.vstack([np.asarray(bboxes)[:,2],np.asarray(bboxes)[:,1],np.asarray(bboxes)[:,4],np.asarray(bboxes)[:,3]]).T
    return bboxes
