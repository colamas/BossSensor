# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from boss_input import IMAGE_SIZE
from boss_train import Dataset, Model


def _make_data(root, count=10):
    for name, value in [('boss', 230), ('other', 20)]:
        os.makedirs(root / name)
        for i in range(count):
            image = np.full((40, 40, 3), value, dtype=np.uint8)
            cv2.imwrite(str(root / name / ('%d.jpg' % i)), image)


def test_dataset_splits_do_not_overlap(tmp_path):
    _make_data(tmp_path)
    dataset = Dataset()
    dataset.read(path=str(tmp_path))

    total = len(dataset.X_train) + len(dataset.X_valid) + len(dataset.X_test)
    assert total == 20
    assert len(dataset.X_train) == 14
    assert dataset.X_train.max() <= 1.0


def test_train_save_load_predict(tmp_path):
    _make_data(tmp_path / 'data')
    dataset = Dataset()
    dataset.read(path=str(tmp_path / 'data'))

    model = Model()
    model.build_model(dataset)
    model.train(dataset, batch_size=4, nb_epoch=1)
    file_path = str(tmp_path / 'store' / 'model.keras')
    model.save(file_path)

    loaded = Model()
    loaded.load(file_path)
    loaded.evaluate(dataset)
    result = loaded.predict(np.zeros((50, 30, 3), dtype=np.uint8))
    assert result in (0, 1)
    result = loaded.predict(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    assert result in (0, 1)
