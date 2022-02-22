import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from scipy.ndimage import rotate



        

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img
    
    
    

def data_aug(train_set, train_labels):
    """perform basic operations of data augmentation on a training set --> return the new trainin
    set and the new training labels

    Args:
        train_set (array of size N*(nx*ny), here 540*300*200)
    """
    rotations = [30, -30, 60, -60, -150, 150, -120, 120]
    n_rot = len(rotations)
    N = train_set.shape[0]
    new_N = N*(n_rot + 2)
    
    new_train_set = np.empty((new_N , 300, 200), dtype=object)
    
    for i in range(N):
        img = train_set[i]
        new_train_set[i*(n_rot + 2)] = img
        for k in range(1, n_rot + 1):
            new_train_set[i*(n_rot + 2) + k] = rotate_img(img, rotations[k-1])
        new_train_set[i*(n_rot + 2) + n_rot + 1] = np.fliplr(img)
    
    new_train_labels = np.repeat(train_labels, n_rot + 2)
        
    return new_train_set, new_train_labels