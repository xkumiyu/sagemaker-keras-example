import argparse
import time
from pathlib import Path

import numpy as np
from sagemaker.tensorflow import TensorFlowPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_name')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--mnist_index', '-i', type=int, default=0)
    args = parser.parse_args()

    predictor = TensorFlowPredictor(args.endpoint_name)

    image = get_mnist_data(args.data_dir, index=args.mnist_index)

    inputs = {'instances': image}
    t = time.time()
    outputs = predictor.predict(inputs)
    print(f'inference time: {(time.time() - t) * 1000:.2f} ms')

    prediction = np.array(outputs['predictions'][0])
    pred_label = np.argmax(prediction)
    pred_confidence = np.max(prediction)

    print(f'prediction: {pred_label} ({pred_confidence * 100:.1f}%)')


def get_mnist_data(data_dir, index=0):
    data_dir = Path(data_dir)
    x = np.load(str(data_dir / 'test.npz'))['image']
    y = np.load(str(data_dir / 'test.npz'))['label']

    image = x[index]
    image = image.reshape(1, 28, 28, 1)
    image = image.astype(np.float32)
    image /= 255

    label = y[index]
    print(f'ground truth: {label}')

    return image


if __name__ == "__main__":
    main()
