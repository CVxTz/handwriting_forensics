import numpy as np
import cv2
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path


def crop_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    crop = img[y:y + h, x:x + w]  # create a cropped region of the gray image
    return crop


def resize_img(img, h=128, w=1024):

    desired_size_h = h
    desired_size_w = w

    old_size = img.shape[:2]

    ratio = min(desired_size_w/old_size[1], desired_size_h/old_size[0])

    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size_w - new_size[1]
    delta_h = desired_size_h - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


if __name__ == "__main__":

    out_path = "../input/clean_images/"

    os.makedirs(out_path, exist_ok=True)

    image_paths = glob("../input/lineImages/**/*.tif", recursive=True)

    for path in tqdm(image_paths):
        name = Path(path).stem

        img = cv2.imread(path)
        crop = crop_white(img)
        crop = resize_img(crop)
        cv2.imwrite(out_path+name+".png", crop)