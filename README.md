# SageMaker Example for Keras

## Requirements

- Python 3.6
- Tensorflow 1.12

## Setting Environment Variable

### Required

``` sh
SAGEMAKER_ROLE='arn:aws:iam::1234567890'
```

### Optional

``` sh
S3_BUCKET_NAME='sagemaker-example'
```

## Dataset Preparation

Download MNIST dataset and upload to S3.

``` sh
python prepare_mnist.py
```

## Training

Create a SageMaker training job.
Then, the job name is `mnist-yyyy-mm-dd-HH-MM`.

``` sh
python create_training_job.py
```

Source code used for training is stored in S3.
And, after training is complete, the output and model are stored in S3.

- source code: s3://bucket-name/training/job-name/source/sourcedir.tar.gz
- output: s3://bucket-name/training/job-name/output/output.tar.gz
- model: s3://bucket-name/training/job-name/output/model.tar.gz

## TODO

- [ ] Deployment
- [ ] Deployment using BYO model
- [ ] Inference
- [ ] Custom Image
