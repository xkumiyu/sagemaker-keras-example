import argparse
import os

from sagemaker.tensorflow.serving import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_data', help='s3 path to model.tar.gz')
    parser.add_argument('--instance_type', default='ml.m5.large')
    parser.add_argument('--role', default=os.environ['SAGEMAKER_ROLE'])
    args = parser.parse_args()

    model = Model(args.model_data, args.role, framework_version='1.12.0')
    model.deploy(initial_instance_count=1, instance_type=args.instance_type)


if __name__ == "__main__":
    main()
