import numpy as np
import cv2 as cv


def resize_pad(img, size, pad_color=0):
    # found in https://stackoverflow.com/a/44724368
    
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv.INTER_AREA
    else: # stretching image
        interp = cv.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color]*3

    # scale and pad
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=pad_color)

    return scaled_img

def coord2xyminmax(box: np.ndarray, to_int: bool = False):
    ymin, ymax = np.min(box[:, 1]), np.max(box[:, 1])
    xmin, xmax = np.min(box[:, 0]), np.max(box[:, 0])
    if to_int:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    return xmin, ymin, xmax, ymax


def xywh2xymm(box):
    x,y,w,h = box
    xmin, ymin, xmax, ymax = x, y, x+w, y+h
    return xmin, ymin, xmax, ymax


def xymm2xywh(box):
    xmin, ymin, xmax, ymax = box
    x,y,w,h = xmin, ymin, xmax-xmin, ymax-ymin
    return x,y,w,h

def pad(box, factor=0.1, to_int=True):
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    wf, hf = w * factor, h * factor
    xmin, ymin, xmax, ymax = xmin - wf, ymin - hf, xmax + wf, ymax + hf
    box_out = [xmin, ymin, xmax, ymax]
    if to_int:
        box_out = [int(xmin), int(ymin), int(xmax), int(ymax)]

    # check if minus set to zero
    for idx, val in enumerate(box_out):
        if val < 0:
            box_out[idx] = 0

    return tuple(box_out)


def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew


