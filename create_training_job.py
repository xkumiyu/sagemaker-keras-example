import argparse
import os
from datetime import datetime as dt

from sagemaker.session import Session
from sagemaker.tensorflow import TensorFlow


def main():
    session = Session()
    s3_bucket_name = os.getenv('S3_BUCKET_NAME', session.default_bucket())

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--role', default=os.environ['SAGEMAKER_ROLE'])
    parser.add_argument('--input_data',
                        default=f's3://{s3_bucket_name}/dataset/mnist')
    parser.add_argument('--output_path',
                        default=f's3://{s3_bucket_name}/training')
    parser.add_argument('--train_instance_type', default='ml.m5.large')
    parser.add_argument('--wait', action='store_true')
    # parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()

    input_data = {'dataset': args.input_data}
    job_name = 'mnist-' + dt.now().strftime('%Y-%m-%d-%H-%M')

    hyperparameters = {'batch_size': args.batch_size, 'epochs': args.epochs}

    metric_definitions = [
        {
            'Name': 'train loss',
            'Regex': r'loss: (\S+)'
        },
        {
            'Name': 'valid loss',
            'Regex': r'val_loss: (\S+)'
        },
    ]
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='src',
                           role=args.role,
                           train_instance_count=1,
                           train_instance_type=args.train_instance_type,
                           train_volume_size=30,
                           train_max_run=86400,
                           output_path=args.output_path,
                           code_location=args.output_path,
                           py_version='py3',
                           framework_version='1.12.0',
                           hyperparameters=hyperparameters,
                           metric_definitions=metric_definitions)
    estimator.fit(input_data, wait=args.wait, job_name=job_name)


if __name__ == "__main__":
    main()
