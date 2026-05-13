# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = 0, 0, 0, 0
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))


def read_image(file_path):
    image = cv2.imread(file_path)
    return resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)


def traverse_dir(path):
    images = []
    labels = []
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):
            sub_images, sub_labels = traverse_dir(abs_path)
            images.extend(sub_images)
            labels.extend(sub_labels)
        elif file_or_dir.endswith('.jpg'):
            image = read_image(abs_path)
            images.append(image)
            labels.append(path)
    return images, labels


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('boss') else 1 for label in labels])
    return images, labels
