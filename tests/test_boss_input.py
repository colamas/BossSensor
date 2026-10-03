# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from boss_input import IMAGE_SIZE, extract_data, resize_with_pad


def _write_image(path, height, width):
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_resize_with_pad_returns_square_image():
    for shape in [(30, 50, 3), (50, 30, 3), (40, 40, 3)]:
        image = np.full(shape, 255, dtype=np.uint8)
        resized = resize_with_pad(image)
        assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_resize_with_pad_pads_short_edge_with_black():
    image = np.full((20, 40, 3), 255, dtype=np.uint8)
    resized = resize_with_pad(image, 40, 40)
    assert (resized[0] == 0).all()
    assert (resized[-1] == 0).all()
    assert (resized[20] == 255).all()


def test_extract_data_labels_by_directory(tmp_path):
    os.makedirs(tmp_path / 'boss')
    os.makedirs(tmp_path / 'other')
    os.makedirs(tmp_path / 'not_boss')
    _write_image(tmp_path / 'boss' / 'a.jpg', 30, 20)
    _write_image(tmp_path / 'boss' / 'b.png', 20, 30)
    _write_image(tmp_path / 'other' / 'c.jpg', 25, 25)
    _write_image(tmp_path / 'not_boss' / 'd.jpg', 25, 25)
    (tmp_path / 'other' / 'readme.txt').write_text('ignored')

    images, labels = extract_data(str(tmp_path))

    assert images.shape == (4, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert sorted(labels.tolist()) == [0, 0, 1, 1]


def test_extract_data_does_not_accumulate_between_calls(tmp_path):
    os.makedirs(tmp_path / 'boss')
    _write_image(tmp_path / 'boss' / 'a.jpg', 30, 30)

    first, _ = extract_data(str(tmp_path))
    second, _ = extract_data(str(tmp_path))

    assert len(first) == len(second) == 1
