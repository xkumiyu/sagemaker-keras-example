import os
from pathlib import Path
import argparse

import numpy as np
from sagemaker.session import Session
from tensorflow.keras.datasets import mnist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data_dir.mkdir()
        np.savez(str(data_dir / 'train'), image=x_train, label=y_train)
        np.savez(str(data_dir / 'test'), image=x_test, label=y_test)

    session = Session()
    s3_bucket_name = os.getenv('S3_BUCKET_NAME', session.default_bucket())
    session.upload_data(path=str(data_dir),
                        bucket=s3_bucket_name,
                        key_prefix='dataset/mnist')


if __name__ == "__main__":
    main()
