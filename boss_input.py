# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # cv2.resize takes (width, height)
    resized_image = cv2.resize(constant, (width, height))

    return resized_image


def traverse_dir(path, images=None, labels=None):
    if images is None:
        images = []
    if labels is None:
        labels = []
    for file_or_dir in sorted(os.listdir(path)):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path, images, labels)
        else:                        # file
            if file_or_dir.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = read_image(abs_path)
                if image is None:
                    print('Skipped unreadable image:', abs_path)
                    continue
                images.append(image)
                labels.append(os.path.normpath(path))

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return None
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if os.path.basename(label) == 'boss' else 1 for label in labels])

    return images, labels
