import numpy as np
from pathlib import Path
import tensorflow as tf

from config import IMAGE_SIZE, NUM_CLASSES


def dataset(data_dir, is_train=True):
    data_dir = Path(data_dir)
    if is_train:
        x = np.load(str(data_dir / 'train.npz'))['image']
        y = np.load(str(data_dir / 'train.npz'))['label']
    else:
        x = np.load(str(data_dir / 'test.npz'))['image']
        y = np.load(str(data_dir / 'test.npz'))['label']

    x = x.reshape(x.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    x = x.astype('float32') / 255
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    return x, y
